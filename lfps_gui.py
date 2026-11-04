"""GUI desktop untuk analisis LFPS pada satu file atau batch MiniSEED."""

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib-cache"))

import matplotlib

# Fungsi LFPS hanya menyimpan gambar. Backend noninteraktif aman dipakai dari
# worker thread dan tidak berkonflik dengan event loop Tkinter.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime, read

from func_clean import (
    calPSD_STALTA_dip,
    calPSD_STALTA_dip_nolog,
    cal_polar_dip,
    calculateVH_STALTA_dip,
    plot_window_dip,
    preprocess,
)


@dataclass(frozen=True)
class LFPSParameters:
    input_folder: Path
    input_file: Path | None
    output_folder: Path
    recursive: bool
    trim_enabled: bool
    trim_start: str
    trim_end: str
    detrend: bool
    bandpass: bool
    spline_order: int
    dspline: int
    preprocess_fmin: float
    preprocess_fmax: float
    window_length: float
    overlap: float
    analysis_fmin: float
    analysis_fmax: float
    plot_fmin: float
    plot_fmax: float
    cft_min: float
    cft_max: float
    dip_threshold: float
    spectral_window: str

    @property
    def shift(self) -> float:
        return self.window_length - self.overlap

    def validate(self) -> None:
        if not self.input_folder.is_dir():
            raise ValueError("Folder input tidak ditemukan.")
        if self.input_file is not None:
            if not self.input_file.is_file():
                raise ValueError("File MiniSEED tidak ditemukan.")
            if self.input_file.suffix.lower() not in {".mseed", ".miniseed"}:
                raise ValueError("File input harus berekstensi .mseed atau .miniseed.")
        if self.trim_enabled:
            if self.input_file is None:
                raise ValueError("Trim waktu hanya tersedia untuk mode satu file.")
            if not self.trim_start or not self.trim_end:
                raise ValueError("Starttime dan endtime trim harus diisi.")
            try:
                trim_start = UTCDateTime(self.trim_start)
                trim_end = UTCDateTime(self.trim_end)
            except Exception as exc:
                raise ValueError("Format starttime atau endtime tidak valid.") from exc
            if trim_end <= trim_start:
                raise ValueError("Endtime trim harus lebih besar dari starttime.")
        if self.window_length <= 0:
            raise ValueError("Panjang window harus lebih besar dari 0.")
        if not 0 <= self.overlap < self.window_length:
            raise ValueError("Overlap harus >= 0 dan lebih kecil dari panjang window.")
        if self.spline_order < 1 or self.dspline < 1:
            raise ValueError("Order spline dan dspline harus berupa bilangan positif.")
        if self.bandpass and not 0 < self.preprocess_fmin < self.preprocess_fmax:
            raise ValueError("Frekuensi bandpass harus memenuhi 0 < min < max.")
        if not 0 < self.analysis_fmin < self.analysis_fmax:
            raise ValueError("Rentang frekuensi analisis harus memenuhi 0 < min < max.")
        if not 0 < self.plot_fmin < self.plot_fmax:
            raise ValueError("Rentang frekuensi plot harus memenuhi 0 < min < max.")
        if self.cft_min >= self.cft_max:
            raise ValueError("CFT minimum harus lebih kecil dari CFT maksimum.")
        if not 0 <= self.dip_threshold <= 90:
            raise ValueError("Ambang dip harus berada dalam rentang 0 sampai 90 derajat.")
        if self.spectral_window not in {"boxcar", "hann", "hamming", "blackman"}:
            raise ValueError("Window spektral harus boxcar, hann, hamming, atau blackman.")


def find_mseed_files(folder: Path, recursive: bool) -> list[Path]:
    """Temukan .mseed/.miniseed tanpa membedakan huruf besar dan kecil."""
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    extensions = {".mseed", ".miniseed"}
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in extensions)


def select_zne(stream: Stream) -> Stream:
    """Susun satu trace Z, N, E agar kontrak fungsi LFPS konsisten."""
    selected = []
    for component in "ZNE":
        traces = stream.select(component=component)
        if not traces:
            raise ValueError(f"Komponen {component} tidak ditemukan.")
        selected.append(traces[0].copy())
    return Stream(traces=selected)


def require_result(name: str, result, expected_length: int):
    """Ubah return pendek dari fungsi LFPS menjadi pesan error yang jelas."""
    if result is None or len(result) != expected_length:
        raise ValueError(
            f"{name} tidak menghasilkan data lengkap; kemungkinan tidak ada window yang lolos."
        )
    return result


def parameter_attributes(params: LFPSParameters) -> dict[str, object]:
    """Parameter terpilih dalam bentuk yang aman untuk Excel dan JSON."""
    return {
        "Param Input Mode": "single_file" if params.input_file else "batch_folder",
        "Param Recursive": params.recursive,
        "Param Trim Enabled": params.trim_enabled,
        "Param Trim Start UTC": params.trim_start if params.trim_enabled else "",
        "Param Trim End UTC": params.trim_end if params.trim_enabled else "",
        "Param Detrend": params.detrend,
        "Param Bandpass": params.bandpass,
        "Param Spline Order": params.spline_order,
        "Param Dspline": params.dspline,
        "Param Bandpass Min (Hz)": params.preprocess_fmin,
        "Param Bandpass Max (Hz)": params.preprocess_fmax,
        "Param Window Length (s)": params.window_length,
        "Param Overlap (s)": params.overlap,
        "Param Shift (s)": params.shift,
        "Param Analysis Fmin (Hz)": params.analysis_fmin,
        "Param Analysis Fmax (Hz)": params.analysis_fmax,
        "Param Plot Fmin (Hz)": params.plot_fmin,
        "Param Plot Fmax (Hz)": params.plot_fmax,
        "Param CFT Min": params.cft_min,
        "Param CFT Max": params.cft_max,
        "Param Dip Threshold (degree)": params.dip_threshold,
        "Param Spectral Window": params.spectral_window,
    }


def write_run_log(path: Path, content: dict[str, object]) -> None:
    """Tulis log atomik agar file tidak tertinggal setengah jika proses terputus."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(content, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def analyse_file(source: Path, params: LFPSParameters) -> dict[str, object]:
    """Jalankan semua plot dan metrik run_batch untuk satu berkas."""
    data = select_zne(read(str(source)))
    data = preprocess(
        data,
        detrend=params.detrend,
        bandpass=params.bandpass,
        order=params.spline_order,
        dspline=params.dspline,
        freqmin=params.preprocess_fmin,
        freqmax=params.preprocess_fmax,
    )

    start_time = max(trace.stats.starttime for trace in data)
    end_time = min(trace.stats.endtime for trace in data)
    if end_time <= start_time:
        raise ValueError("Ketiga komponen tidak memiliki rentang waktu yang beririsan.")

    if params.trim_enabled:
        requested_start = UTCDateTime(params.trim_start)
        requested_end = UTCDateTime(params.trim_end)
        if requested_start < start_time or requested_end > end_time:
            raise ValueError(
                "Rentang trim berada di luar irisan waktu ketiga komponen "
                f"({start_time} sampai {end_time})."
            )
        start_time, end_time = requested_start, requested_end
    data.trim(start_time, end_time)

    duration = float(end_time - start_time)
    if duration <= params.window_length:
        raise ValueError("Durasi rekaman tidak lebih panjang dari panjang window.")

    relative = source.relative_to(params.input_folder)
    station = source.stem
    station_output = params.output_folder / relative.parent / station
    station_output.mkdir(parents=True, exist_ok=True)
    save_base = str(station_output / station)

    common = dict(
        fmin=params.analysis_fmin,
        fmax=params.analysis_fmax,
        fmin_plot=params.plot_fmin,
        fmax_plot=params.plot_fmax,
        cft_min=params.cft_min,
        cft_max=params.cft_max,
        dip_tres=params.dip_threshold,
    )

    quality = require_result(
        "plot_window_dip",
        plot_window_dip(data, params.window_length, params.shift, save_base, **common),
        7,
    )
    quality_pct, status, quality_2h, status_2h, rms_z, rms_n, rms_e = quality

    psd_log = require_result(
        "calPSD_STALTA_dip",
        calPSD_STALTA_dip(
            data, params.window_length, params.shift, save_base,
            spectral_window=params.spectral_window, **common
        ),
        7,
    )
    psd_max_log, psd_freq_log, psd_integral_log, *_ = psd_log

    psd_linear = require_result(
        "calPSD_STALTA_dip_nolog",
        calPSD_STALTA_dip_nolog(
            data, params.window_length, params.shift, save_base,
            spectral_window=params.spectral_window, **common
        ),
        7,
    )
    psd_max, psd_freq, psd_integral, *_ = psd_linear

    vh_result = require_result(
        "calculateVH_STALTA_dip",
        calculateVH_STALTA_dip(
            data, params.window_length, params.shift, save_base,
            spectral_window=params.spectral_window, **common
        ),
        6,
    )
    vh_max, vh_frequency, *_ = vh_result

    polar = require_result(
        "cal_polar_dip",
        cal_polar_dip(data, params.window_length, params.shift, save_base, **common),
        2,
    )
    (rect, dip, azimuth, eigen), (rect_mean, dip_mean, azimuth_mean, eigen_mean) = polar

    # Fungsi LFPS memakai kondisi ``t1 < t_end - windows``.
    total_window = int(np.ceil((duration - params.window_length) / params.shift))
    accepted_window = int(round((quality_pct / 100.0) * total_window))

    return {
        "File": str(relative),
        "Station": station,
        "Network Codes": ", ".join(sorted({trace.stats.network for trace in data})),
        "Station Codes": ", ".join(sorted({trace.stats.station for trace in data})),
        "Location Codes": ", ".join(sorted({trace.stats.location for trace in data})),
        "Channels": ", ".join(trace.stats.channel for trace in data),
        "Start Time": str(start_time),
        "End Time": str(end_time),
        "Trim Applied": params.trim_enabled,
        "Total Samples": int(data[0].stats.npts),
        "Sampling Rate (Hz)": float(data[0].stats.sampling_rate),
        "Duration (s)": duration,
        "Total Window": total_window,
        "Window Lolos": accepted_window,
        "Quality (%)": float(quality_pct),
        "Status": status,
        "Quality 2h (%)": float(quality_2h),
        "Status 2h": status_2h,
        "Spectral Window": params.spectral_window,
        "PSD Z Max": float(psd_max),
        "Freq PSD Z": float(psd_freq),
        "PSD IZ": float(psd_integral),
        "PSD Z Max (log)": float(psd_max_log),
        "Freq PSD Z (log)": float(psd_freq_log),
        "PSD IZ (log)": float(psd_integral_log),
        "VH Max": float(vh_max),
        "Freq VH": float(vh_frequency),
        "Rect (All)": float(rect),
        "Dip (All)": float(dip),
        "Azimuth (All)": float(azimuth),
        "Eigen (All)": float(eigen),
        "Rect (Mean)": float(rect_mean),
        "Dip (Mean)": float(dip_mean),
        "Azimuth (Mean)": float(azimuth_mean),
        "Eigen (Mean)": float(eigen_mean),
        "Az_rms": float(rms_z),
        "An_rms": float(rms_n),
        "Ae_rms": float(rms_e),
        "Output Folder": str(station_output),
        "Error": "",
    }


def process_batch(params: LFPSParameters, emit, cancelled) -> tuple[Path, int, int]:
    """Proses semua berkas dan kirim event log/progress ke GUI."""
    params.validate()
    files = ([params.input_file] if params.input_file is not None
             else find_mseed_files(params.input_folder, params.recursive))
    if not files:
        raise ValueError("Tidak ditemukan berkas .mseed atau .miniseed pada folder input.")

    params.output_folder.mkdir(parents=True, exist_ok=True)
    log_path = params.output_folder / "LFPS_Run_Log.json"
    started_at = datetime.now(timezone.utc)
    run_log: dict[str, object] = {
        "started_at_utc": started_at.isoformat(),
        "status": "running",
        "input": {
            "mode": "single_file" if params.input_file else "batch_folder",
            "folder": str(params.input_folder),
            "file": str(params.input_file) if params.input_file else None,
            "output_folder": str(params.output_folder),
        },
        "parameters": parameter_attributes(params),
        "files": [],
    }
    write_run_log(log_path, run_log)
    rows: list[dict[str, object]] = []
    errors = 0
    emit("total", len(files))

    for index, source in enumerate(files, start=1):
        if cancelled():
            emit("log", "Proses dibatalkan. Menyimpan hasil yang sudah selesai...")
            break
        emit("log", f"[{index}/{len(files)}] Memproses {source.name}")
        try:
            row = analyse_file(source, params)
            emit("log", f"  Selesai: {source.name}")
        except Exception as exc:
            errors += 1
            row = {
                "File": str(source.relative_to(params.input_folder)),
                "Station": source.stem,
                "Error": f"{type(exc).__name__}: {exc}",
            }
            plt.close("all")
            emit("log", f"  Gagal: {type(exc).__name__}: {exc}")
        row.update(parameter_attributes(params))
        rows.append(row)
        run_log["files"].append(row)
        run_log["processed_files"] = len(rows)
        run_log["successful_files"] = len(rows) - errors
        run_log["failed_files"] = errors
        write_run_log(log_path, run_log)
        emit("progress", index)

    if not rows:
        raise RuntimeError("Belum ada hasil yang dapat disimpan.")

    run_log["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_log["status"] = ("cancelled" if cancelled()
                         else "completed_with_errors" if errors
                         else "completed")
    write_run_log(log_path, run_log)

    dataframe = pd.DataFrame(rows)
    excel_path = params.output_folder / "Hasil_Akhir_LFPS.xlsx"
    try:
        dataframe.to_excel(excel_path, index=False)
    except ImportError as exc:
        csv_path = params.output_folder / "Hasil_Akhir_LFPS.csv"
        dataframe.to_csv(csv_path, index=False)
        emit("log", f"Engine Excel tidak tersedia ({exc}). Hasil disimpan sebagai CSV.")
        summary_path = csv_path
    else:
        summary_path = excel_path

    run_log["summary_file"] = str(summary_path)
    write_run_log(log_path, run_log)
    emit("log", f"Log parameter dan atribut disimpan: {log_path}")
    return summary_path, len(rows) - errors, errors


class LFPSApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("LFPS Processor")
        self.geometry("940x780")
        self.minsize(820, 680)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.variables: dict[str, tk.Variable] = {}
        self._build_ui()
        self.after(100, self._poll_events)

    def _variable(self, name: str, value, cls=tk.StringVar):
        variable = cls(value=value)
        self.variables[name] = variable
        return variable

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)

        folders = ttk.LabelFrame(outer, text="Input dan output", padding=10)
        folders.pack(fill="x")

        ttk.Label(folders, text="Mode input").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        mode_frame = ttk.Frame(folders)
        mode_frame.grid(row=0, column=1, columnspan=2, sticky="w")
        input_mode = self._variable("input_mode", "folder")
        ttk.Radiobutton(
            mode_frame, text="Batch folder", value="folder", variable=input_mode,
            command=self._update_input_mode,
        ).pack(side="left")
        ttk.Radiobutton(
            mode_frame, text="Satu file", value="file", variable=input_mode,
            command=self._update_input_mode,
        ).pack(side="left", padx=(14, 0))

        self.input_label = ttk.Label(folders, text="Folder MiniSEED")
        self.input_label.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(folders, textvariable=self._variable("input_path", "")).grid(
            row=1, column=1, sticky="ew", pady=3
        )
        ttk.Button(folders, text="Pilih...", command=self._choose_input).grid(
            row=1, column=2, padx=(8, 0), pady=3
        )
        self._path_row(folders, 2, "Folder output", "output_folder", self._choose_output)
        self.recursive_check = ttk.Checkbutton(
            folders,
            text="Cari MiniSEED di subfolder",
            variable=self._variable("recursive", False, tk.BooleanVar),
        )
        self.recursive_check.grid(row=3, column=1, sticky="w", pady=(5, 0))

        self.trim_check = ttk.Checkbutton(
            folders,
            text="Trim data berdasarkan waktu UTC",
            variable=self._variable("trim_enabled", False, tk.BooleanVar),
            command=self._toggle_trim,
        )
        self.trim_check.grid(row=4, column=1, sticky="w", pady=(5, 0))
        trim_frame = ttk.Frame(folders)
        trim_frame.grid(row=5, column=1, columnspan=2, sticky="ew", pady=(3, 0))
        ttk.Label(trim_frame, text="Starttime UTC (ISO)").grid(row=0, column=0, sticky="w")
        self.trim_start_entry = ttk.Entry(
            trim_frame, textvariable=self._variable("trim_start", ""), width=28
        )
        self.trim_start_entry.grid(row=0, column=1, sticky="ew", padx=(6, 14))
        ttk.Label(trim_frame, text="Endtime UTC (ISO)").grid(row=0, column=2, sticky="w")
        self.trim_end_entry = ttk.Entry(
            trim_frame, textvariable=self._variable("trim_end", ""), width=28
        )
        self.trim_end_entry.grid(row=0, column=3, sticky="ew", padx=(6, 0))
        trim_frame.columnconfigure(1, weight=1)
        trim_frame.columnconfigure(3, weight=1)
        folders.columnconfigure(1, weight=1)
        self._update_input_mode()

        parameters = ttk.LabelFrame(outer, text="Parameter LFPS", padding=10)
        parameters.pack(fill="x", pady=(10, 0))

        defaults = [
            ("Window (detik)", "window_length", "60"),
            ("Overlap (detik)", "overlap", "30"),
            ("Frekuensi analisis min (Hz)", "analysis_fmin", "1"),
            ("Frekuensi analisis max (Hz)", "analysis_fmax", "6"),
            ("Frekuensi plot min (Hz)", "plot_fmin", "0.5"),
            ("Frekuensi plot max (Hz)", "plot_fmax", "10"),
            ("CFT minimum", "cft_min", "0.2"),
            ("CFT maksimum", "cft_max", "2"),
            ("Ambang dip (derajat)", "dip_threshold", "0"),
            ("Bandpass min (Hz)", "preprocess_fmin", "1"),
            ("Bandpass max (Hz)", "preprocess_fmax", "10"),
            ("Order spline", "spline_order", "3"),
            ("Dspline", "dspline", "500"),
        ]
        for index, (label, name, default) in enumerate(defaults):
            row, pair = divmod(index, 2)
            column = pair * 2
            ttk.Label(parameters, text=label).grid(row=row, column=column, sticky="w", padx=(0, 6), pady=3)
            ttk.Entry(parameters, textvariable=self._variable(name, default), width=15).grid(
                row=row, column=column + 1, sticky="ew", padx=(0, 18), pady=3
            )
        check_row = (len(defaults) + 1) // 2
        ttk.Label(parameters, text="Window spektral").grid(
            row=check_row, column=0, sticky="w", padx=(0, 6), pady=3
        )
        ttk.Combobox(
            parameters,
            textvariable=self._variable("spectral_window", "boxcar"),
            values=("boxcar", "hann", "hamming", "blackman"),
            state="readonly",
            width=13,
        ).grid(row=check_row, column=1, sticky="ew", padx=(0, 18), pady=3)
        ttk.Checkbutton(
            parameters,
            text="Detrend spline",
            variable=self._variable("detrend", True, tk.BooleanVar),
        ).grid(row=check_row + 1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            parameters,
            text="Bandpass",
            variable=self._variable("bandpass", True, tk.BooleanVar),
        ).grid(row=check_row + 1, column=2, columnspan=2, sticky="w", pady=(6, 0))
        parameters.columnconfigure(1, weight=1)
        parameters.columnconfigure(3, weight=1)

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(10, 0))
        self.start_button = ttk.Button(controls, text="Jalankan LFPS", command=self._start)
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(controls, text="Batalkan", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=(8, 0))
        self.open_button = ttk.Button(controls, text="Buka folder output", command=self._open_output)
        self.open_button.pack(side="left", padx=(8, 0))
        self.status_var = tk.StringVar(value="Siap")
        ttk.Label(controls, textvariable=self.status_var).pack(side="right")

        self.progress = ttk.Progressbar(outer, mode="determinate")
        self.progress.pack(fill="x", pady=(10, 0))

        log_frame = ttk.LabelFrame(outer, text="Log", padding=6)
        log_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.log = tk.Text(log_frame, height=12, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _path_row(self, parent, row: int, label: str, name: str, command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(parent, textvariable=self._variable(name, "")).grid(
            row=row, column=1, sticky="ew", pady=3
        )
        ttk.Button(parent, text="Pilih...", command=command).grid(row=row, column=2, padx=(8, 0), pady=3)

    def _choose_input(self) -> None:
        if self.variables["input_mode"].get() == "file":
            selected = filedialog.askopenfilename(
                title="Pilih satu file MiniSEED",
                filetypes=(("MiniSEED", "*.mseed *.miniseed"), ("Semua file", "*.*")),
            )
        else:
            selected = filedialog.askdirectory(title="Pilih folder yang berisi MiniSEED")
        if selected:
            self.variables["input_path"].set(selected)
            if not self.variables["output_folder"].get():
                base_folder = Path(selected).parent if Path(selected).is_file() else Path(selected)
                self.variables["output_folder"].set(str(base_folder / "LFPS_Output"))
            if self.variables["input_mode"].get() == "file":
                self._populate_file_times(Path(selected))

    def _populate_file_times(self, source: Path) -> None:
        """Isi rentang trim dari irisan waktu Z/N/E pada file terpilih."""
        try:
            stream = select_zne(read(str(source), headonly=True))
            start_time = max(trace.stats.starttime for trace in stream)
            end_time = min(trace.stats.endtime for trace in stream)
            self.variables["trim_start"].set(str(start_time))
            self.variables["trim_end"].set(str(end_time))
        except Exception as exc:
            self.variables["trim_start"].set("")
            self.variables["trim_end"].set("")
            messagebox.showwarning(
                "Waktu MiniSEED",
                f"Rentang waktu file tidak dapat dibaca: {exc}",
                parent=self,
            )

    def _update_input_mode(self) -> None:
        single_file = self.variables["input_mode"].get() == "file"
        self.input_label.configure(text="File MiniSEED" if single_file else "Folder MiniSEED")
        self.recursive_check.configure(state="disabled" if single_file else "normal")
        self.trim_check.configure(state="normal" if single_file else "disabled")
        if single_file:
            self.variables["recursive"].set(False)
        else:
            self.variables["trim_enabled"].set(False)
        self._toggle_trim()

    def _toggle_trim(self) -> None:
        enabled = (
            self.variables["input_mode"].get() == "file"
            and bool(self.variables["trim_enabled"].get())
        )
        state = "normal" if enabled else "disabled"
        self.trim_start_entry.configure(state=state)
        self.trim_end_entry.configure(state=state)

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(title="Pilih folder output")
        if selected:
            self.variables["output_folder"].set(selected)

    def _parameters(self) -> LFPSParameters:
        input_text = str(self.variables["input_path"].get()).strip()
        output_text = str(self.variables["output_folder"].get()).strip()
        if not input_text:
            raise ValueError("Folder atau file input belum dipilih.")
        if not output_text:
            raise ValueError("Folder output belum dipilih.")

        def number(name: str) -> float:
            try:
                return float(self.variables[name].get())
            except ValueError as exc:
                raise ValueError(f"Nilai {name} bukan angka yang valid.") from exc

        input_path = Path(input_text).expanduser().resolve()
        single_file = self.variables["input_mode"].get() == "file"

        return LFPSParameters(
            input_folder=input_path.parent if single_file else input_path,
            input_file=input_path if single_file else None,
            output_folder=Path(output_text).expanduser().resolve(),
            recursive=bool(self.variables["recursive"].get()),
            trim_enabled=bool(self.variables["trim_enabled"].get()),
            trim_start=str(self.variables["trim_start"].get()).strip(),
            trim_end=str(self.variables["trim_end"].get()).strip(),
            detrend=bool(self.variables["detrend"].get()),
            bandpass=bool(self.variables["bandpass"].get()),
            spline_order=int(number("spline_order")),
            dspline=int(number("dspline")),
            preprocess_fmin=number("preprocess_fmin"),
            preprocess_fmax=number("preprocess_fmax"),
            window_length=number("window_length"),
            overlap=number("overlap"),
            analysis_fmin=number("analysis_fmin"),
            analysis_fmax=number("analysis_fmax"),
            plot_fmin=number("plot_fmin"),
            plot_fmax=number("plot_fmax"),
            cft_min=number("cft_min"),
            cft_max=number("cft_max"),
            dip_threshold=number("dip_threshold"),
            spectral_window=str(self.variables["spectral_window"].get()).lower(),
        )

    def _start(self) -> None:
        try:
            params = self._parameters()
            params.validate()
        except Exception as exc:
            messagebox.showerror("Parameter tidak valid", str(exc), parent=self)
            return

        self.cancel_event.clear()
        self.progress.configure(value=0, maximum=1)
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status_var.set("Memproses...")
        self._append_log(f"Input: {params.input_file or params.input_folder}")
        if params.trim_enabled:
            self._append_log(f"Trim UTC: {params.trim_start} sampai {params.trim_end}")
        self._append_log(f"Output: {params.output_folder}")
        self.worker = threading.Thread(target=self._run_worker, args=(params,), daemon=True)
        self.worker.start()

    def _run_worker(self, params: LFPSParameters) -> None:
        def emit(kind: str, value) -> None:
            self.events.put((kind, value))

        try:
            result = process_batch(params, emit, self.cancel_event.is_set)
            self.events.put(("done", result))
        except Exception as exc:
            self.events.put(("fatal", (exc, traceback.format_exc())))

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self.status_var.set("Membatalkan setelah file aktif selesai...")

    def _poll_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self._append_log(str(value))
                elif kind == "total":
                    self.progress.configure(maximum=int(value), value=0)
                elif kind == "progress":
                    self.progress.configure(value=int(value))
                elif kind == "done":
                    path, success, errors = value
                    self._finish()
                    self.status_var.set(f"Selesai: {success} berhasil, {errors} gagal")
                    self._append_log(f"Ringkasan disimpan: {path}")
                    messagebox.showinfo(
                        "LFPS selesai",
                        f"Berhasil: {success}\nGagal: {errors}\n\nRingkasan:\n{path}",
                        parent=self,
                    )
                elif kind == "fatal":
                    exc, trace = value
                    self._finish()
                    self.status_var.set("Gagal")
                    self._append_log(trace)
                    messagebox.showerror("LFPS gagal", str(exc), parent=self)
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _finish(self) -> None:
        self.start_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.worker = None

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _open_output(self) -> None:
        raw = str(self.variables["output_folder"].get()).strip()
        if not raw:
            messagebox.showwarning("Folder output", "Pilih folder output terlebih dahulu.", parent=self)
            return
        folder = Path(raw)
        if not folder.exists():
            messagebox.showwarning("Folder output", "Folder output belum dibuat.", parent=self)
            return
        os.startfile(folder)  # Windows


def main() -> None:
    app = LFPSApp()
    app.mainloop()


if __name__ == "__main__":
    main()
