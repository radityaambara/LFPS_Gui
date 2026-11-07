"""GUI desktop untuk analisis LFPS pada satu file atau batch MiniSEED."""

from __future__ import annotations

import queue
import threading
import traceback
import itertools
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
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime, read

from func_clean import (
    calPSD_STALTA_dip,
    calPSD_STALTA_dip_nolog,
    cal_polar_dip,
    build_pair_matrix,
    compare_component_traces,
    calculateVH_STALTA_dip,
    estimate_window_counts,
    normalization_factor,
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
    detrend_method: str
    bandpass: bool
    spline_order: int
    spline_spacing_seconds: float
    taper: bool
    taper_percentage: float
    zero_phase: bool
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
        if self.detrend_method not in {"none", "demean", "linear", "demean_linear", "spline"}:
            raise ValueError("Metode detrend tidak valid.")
        if self.detrend_method == "spline" and not 1 <= self.spline_order <= 5:
            raise ValueError("Order spline harus berada di antara 1 dan 5.")
        if self.detrend_method == "spline" and self.spline_spacing_seconds <= 0:
            raise ValueError("Jarak node spline harus lebih besar dari nol detik.")
        if not 0 <= self.taper_percentage <= 0.5:
            raise ValueError("Persentase taper harus berada antara 0 dan 0.5.")
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


@dataclass(frozen=True)
class CorrelationParameters:
    input_folder: Path
    output_folder: Path
    recursive: bool
    cut_enabled: bool
    cut_start: str
    cut_end: str
    reference_name: str
    min_overlap: float
    max_lag: float
    detrend_method: str
    bandpass: bool
    spline_order: int
    spline_spacing_seconds: float
    taper: bool
    taper_percentage: float
    zero_phase: bool
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
            raise ValueError("Folder MiniSEED korelasi tidak ditemukan.")
        if self.cut_enabled:
            try:
                start, end = UTCDateTime(self.cut_start), UTCDateTime(self.cut_end)
            except Exception as exc:
                raise ValueError("Format waktu cut korelasi tidak valid.") from exc
            if end <= start:
                raise ValueError("Endtime cut harus lebih besar dari starttime.")
        if self.min_overlap <= 0 or self.max_lag < 0:
            raise ValueError("Minimum overlap harus > 0 dan max lag harus >= 0.")
        if self.window_length <= 0 or not 0 <= self.overlap < self.window_length:
            raise ValueError("Window harus > 0 dan overlap harus lebih kecil dari window.")
        if self.detrend_method not in {"none", "demean", "linear", "demean_linear", "spline"}:
            raise ValueError("Metode detrend korelasi tidak valid.")
        if self.detrend_method == "spline" and not 1 <= self.spline_order <= 5:
            raise ValueError("Order spline harus berada di antara 1 dan 5.")
        if self.detrend_method == "spline" and self.spline_spacing_seconds <= 0:
            raise ValueError("Jarak node spline harus lebih besar dari nol detik.")
        if not 0 <= self.taper_percentage <= 0.5:
            raise ValueError("Persentase taper harus berada antara 0 dan 0.5.")
        if self.bandpass and not 0 < self.preprocess_fmin < self.preprocess_fmax:
            raise ValueError("Rentang bandpass korelasi tidak valid.")
        if not 0 < self.analysis_fmin < self.analysis_fmax:
            raise ValueError("Rentang frekuensi analisis korelasi tidak valid.")
        if not 0 < self.plot_fmin < self.plot_fmax:
            raise ValueError("Rentang frekuensi plot korelasi tidak valid.")
        if self.cft_min >= self.cft_max or not 0 <= self.dip_threshold <= 90:
            raise ValueError("CFT atau ambang dip korelasi tidak valid.")


def correlation_attributes(params: CorrelationParameters) -> dict[str, object]:
    return {
        "recursive": params.recursive, "cut_enabled": params.cut_enabled,
        "cut_start_utc": params.cut_start if params.cut_enabled else "",
        "cut_end_utc": params.cut_end if params.cut_enabled else "",
        "reference": params.reference_name or "otomatis (file pertama)",
        "min_overlap_seconds": params.min_overlap, "max_lag_seconds": params.max_lag,
        "detrend_method": params.detrend_method, "bandpass": params.bandpass,
        "spline_order": params.spline_order,
        "spline_spacing_seconds": params.spline_spacing_seconds,
        "taper": params.taper, "taper_percentage": params.taper_percentage,
        "zero_phase_bandpass": params.zero_phase,
        "bandpass_min_hz": params.preprocess_fmin, "bandpass_max_hz": params.preprocess_fmax,
        "window_seconds": params.window_length, "overlap_seconds": params.overlap,
        "analysis_fmin_hz": params.analysis_fmin, "analysis_fmax_hz": params.analysis_fmax,
        "plot_fmin_hz": params.plot_fmin, "plot_fmax_hz": params.plot_fmax,
        "cft_min": params.cft_min, "cft_max": params.cft_max,
        "dip_threshold_degree": params.dip_threshold,
        "spectral_window": params.spectral_window,
    }


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
        "Param Detrend Method": params.detrend_method,
        "Param Bandpass": params.bandpass,
        "Param Spline Order": params.spline_order,
        "Param Spline Spacing (s)": params.spline_spacing_seconds,
        "Param Taper": params.taper,
        "Param Taper Percentage": params.taper_percentage,
        "Param Zero Phase Bandpass": params.zero_phase,
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
        bandpass=params.bandpass,
        order=params.spline_order,
        freqmin=params.preprocess_fmin,
        freqmax=params.preprocess_fmax,
        detrend_method=params.detrend_method,
        spline_spacing_seconds=params.spline_spacing_seconds,
        taper=params.taper,
        taper_percentage=params.taper_percentage,
        zero_phase=params.zero_phase,
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
    total_window, accepted_window = estimate_window_counts(
        duration, params.window_length, params.shift, quality_pct
    )

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


def _safe_name(path: Path) -> str:
    """Nama unik untuk output, termasuk nama subfolder bila mode rekursif."""
    return "__".join(path.with_suffix("").parts).replace(" ", "_")


def _plot_matrix(matrix: pd.DataFrame, title: str, label: str, path: Path,
                 vmin=None, vmax=None, cmap="coolwarm") -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(matrix) * 0.8), max(6, len(matrix) * 0.7)))
    image = ax.imshow(matrix.values, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(image, ax=ax, label=label)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = matrix.iat[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:.3g}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def process_correlation(params: CorrelationParameters, emit, cancelled) -> tuple[Path, int, int]:
    """Bandingkan waveform, PSD, dan V/H beberapa alat pada waktu yang sama."""
    params.validate()
    files = find_mseed_files(params.input_folder, params.recursive)
    if len(files) < 2:
        raise ValueError("Korelasi membutuhkan sedikitnya dua file MiniSEED.")

    root = params.output_folder / "Correlation_Comparison"
    pair_dir, heatmap_dir = root / "pair", root / "heatmap"
    psd_dir, vh_dir = root / "psd", root / "vhsr"
    for folder in (root, pair_dir, heatmap_dir, psd_dir, vh_dir):
        folder.mkdir(parents=True, exist_ok=True)
    log_path = root / "Correlation_Run_Log.json"
    run_log: dict[str, object] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "input_folder": str(params.input_folder), "output_folder": str(root),
        "parameters": correlation_attributes(params), "files": [], "errors": [],
    }
    write_run_log(log_path, run_log)
    emit("corr_total", len(files) + (len(files) * (len(files) - 1) // 2) * 3)

    loaded: list[dict[str, object]] = []
    errors = 0
    for index, source in enumerate(files, 1):
        if cancelled():
            break
        emit("corr_log", f"[{index}/{len(files)}] Membaca {source.name}")
        try:
            stream = select_zne(read(str(source)))
            stream = preprocess(
                stream, bandpass=params.bandpass, order=params.spline_order,
                freqmin=params.preprocess_fmin, freqmax=params.preprocess_fmax,
                detrend_method=params.detrend_method,
                spline_spacing_seconds=params.spline_spacing_seconds,
                taper=params.taper, taper_percentage=params.taper_percentage,
                zero_phase=params.zero_phase,
            )
            start = max(trace.stats.starttime for trace in stream)
            end = min(trace.stats.endtime for trace in stream)
            if params.cut_enabled:
                start, end = max(start, UTCDateTime(params.cut_start)), min(end, UTCDateTime(params.cut_end))
            if end <= start:
                raise ValueError("Tidak beririsan dengan rentang cut.")
            stream.trim(start, end)
            relative = source.relative_to(params.input_folder)
            loaded.append({"name": _safe_name(relative), "file": str(relative),
                           "stream": stream, "start": start, "end": end})
            run_log["files"].append({"file": str(relative), "start": str(start), "end": str(end)})
        except Exception as exc:
            errors += 1
            message = f"{source.name}: {type(exc).__name__}: {exc}"
            run_log["errors"].append(message)
            emit("corr_log", f"  Gagal: {message}")
        emit("corr_progress", index)
        write_run_log(log_path, run_log)
    if len(loaded) < 2:
        raise ValueError("Kurang dari dua file berhasil dibaca untuk korelasi.")

    requested_ref = params.reference_name.strip().lower()
    reference = next((item for item in loaded if requested_ref and
                      (requested_ref == str(item["name"]).lower() or
                       requested_ref == Path(str(item["file"])).stem.lower())), loaded[0])
    run_log["reference_used"] = reference["name"]

    # Perbandingan bentuk spektrum dan faktor gain terhadap alat referensi.
    spectra: list[dict[str, object]] = []
    group_psd, ax_psd = plt.subplots(figsize=(12, 7))
    group_vh, ax_vh = plt.subplots(figsize=(12, 7))
    for item in loaded:
        if cancelled():
            break
        try:
            common = dict(cft_min=params.cft_min, cft_max=params.cft_max,
                          dip_tres=params.dip_threshold, fmin=params.analysis_fmin,
                          fmax=params.analysis_fmax, fmin_plot=params.plot_fmin,
                          fmax_plot=params.plot_fmax, spectral_window=params.spectral_window)
            psd = require_result("PSD", calPSD_STALTA_dip_nolog(
                item["stream"], params.window_length, params.shift,
                str(psd_dir / str(item["name"])), **common), 7)
            vh = require_result("V/H", calculateVH_STALTA_dip(
                item["stream"], params.window_length, params.shift,
                str(vh_dir / str(item["name"])), **common), 6)
            ax_psd.plot(psd[5], psd[6], label=item["name"])
            ax_vh.plot(vh[4], vh[5], label=item["name"])
            spectra.append({"name": item["name"], "psd_max": float(psd[0]),
                            "psd_peak_hz": float(psd[1]), "psd_integral": float(psd[2]),
                            "vh_max": float(vh[0]), "vh_peak_hz": float(vh[1])})
        except Exception as exc:
            errors += 1
            message = f"Spektrum {item['name']}: {type(exc).__name__}: {exc}"
            run_log["errors"].append(message)
            emit("corr_log", f"  {message}")
    for fig, ax, ylabel, filename in (
        (group_psd, ax_psd, "PSD-Z", "GROUP_PSDZ.png"),
        (group_vh, ax_vh, "V/H", "GROUP_VH.png")):
        ax.set(xlabel="Frekuensi (Hz)", ylabel=ylabel); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig((psd_dir if "PSD" in filename else vh_dir) / filename, dpi=200)
        plt.close(fig)
    if spectra:
        ref_spec = next((row for row in spectra if row["name"] == reference["name"]), spectra[0])
        for row in spectra:
            row["psd_gain_to_reference"] = normalization_factor(
                ref_spec["psd_max"], row["psd_max"]
            )
            row["vh_gain_to_reference"] = normalization_factor(
                ref_spec["vh_max"], row["vh_max"]
            )
        pd.DataFrame(spectra).to_csv(root / "REKAP_NORMALIZATION.csv", index=False)

    rows: list[dict[str, object]] = []
    pair_count = len(loaded) * (len(loaded) - 1) // 2
    step = len(files)
    for left, right in itertools.combinations(loaded, 2):
        start, end = max(left["start"], right["start"]), min(left["end"], right["end"])
        if end - start < params.min_overlap:
            emit("corr_log", f"Lewati {left['name']} vs {right['name']}: overlap terlalu pendek")
            step += 3; emit("corr_progress", step); continue
        for component in "ZNE":
            if cancelled():
                break
            try:
                tr1 = left["stream"].select(component=component)[0].copy().trim(start, end)
                tr2 = right["stream"].select(component=component)[0].copy().trim(start, end)
                metrics = compare_component_traces(tr1, tr2, params.max_lag)
                pair_name = f"{left['name']}_VS_{right['name']}_{component}"
                figure_path = pair_dir / f"{pair_name}.png"
                fig, ax = plt.subplots(figsize=(14, 4))
                times = tr1.stats.starttime.matplotlib_date + metrics["time_seconds"] / 86400.0
                ax.plot(times, metrics["normalized_1"] + 1.2, lw=.6, label=left["name"])
                ax.plot(times, metrics["normalized_2"] - 1.2, lw=.6, label=right["name"])
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
                ax.set_title(
                    f"{component}: corr={metrics['correlation_max']:.4f}, "
                    f"lag={metrics['lag_seconds']:.4f} s"
                )
                ax.set_xlabel("Waktu UTC"); ax.grid(alpha=.3); ax.legend(); fig.autofmt_xdate()
                fig.tight_layout(); fig.savefig(figure_path, dpi=200); plt.close(fig)
                rows.append({"component": component, "alat_1": left["name"],
                             "alat_2": right["name"], "start_overlap": str(start),
                             "end_overlap": str(end), "duration_seconds": float(end-start),
                             "sampling_rate_hz": metrics["sampling_rate_hz"],
                             "correlation_max": metrics["correlation_max"],
                             "lag_seconds": metrics["lag_seconds"],
                             "pearson_zero_lag": metrics["pearson_zero_lag"],
                             "rms_difference_raw": metrics["rms_difference_raw"],
                             "rms_difference_normalized": metrics["rms_difference_normalized"],
                             "plot": str(figure_path)})
            except Exception as exc:
                errors += 1
                run_log["errors"].append(f"{left['name']} vs {right['name']} {component}: {exc}")
            step += 1; emit("corr_progress", step)
    if not rows:
        raise RuntimeError("Tidak ada pasangan dengan overlap yang cukup untuk dihitung.")

    result = pd.DataFrame(rows)
    csv_path = root / "REKAP_XCORR_FINAL.csv"
    result.to_csv(csv_path, index=False)
    names = [str(item["name"]) for item in loaded]
    for component in "ZNE":
        subset = result[result.component == component]
        for column, title, label, limits in (
            ("correlation_max", "Korelasi maksimum", "Koefisien", (-1, 1)),
            ("lag_seconds", "Lag saat korelasi maksimum", "Lag (detik)", (None, None)),
            ("rms_difference_normalized", "RMS selisih ternormalisasi", "RMS", (0, None))):
            diagonal = 1.0 if column == "correlation_max" else 0.0
            matrix = pd.DataFrame(
                build_pair_matrix(
                    names, subset.to_dict("records"), column, diagonal=diagonal,
                    antisymmetric=column == "lag_seconds",
                ),
                index=names,
                columns=names,
            )
            _plot_matrix(matrix, f"{title} - {component}", label,
                         heatmap_dir / f"{component}_{column}.png", *limits)

    run_log.update({"ended_at_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "cancelled" if cancelled() else "completed_with_errors" if errors else "completed",
                    "loaded_files": len(loaded), "pair_count": pair_count,
                    "result_rows": len(rows), "result_csv": str(csv_path)})
    write_run_log(log_path, run_log)
    emit("corr_log", f"Referensi: {reference['name']}")
    emit("corr_log", f"Hasil korelasi: {csv_path}")
    return csv_path, len(rows), errors


class LFPSApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("LFPS Processor")
        self.geometry("940x780")
        self.minsize(820, 680)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.corr_cancel_event = threading.Event()
        self.corr_worker: threading.Thread | None = None
        self.variables: dict[str, tk.Variable] = {}
        self._build_ui()
        self.after(100, self._poll_events)

    def _variable(self, name: str, value, cls=tk.StringVar):
        variable = cls(value=value)
        self.variables[name] = variable
        return variable

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        outer = ttk.Frame(notebook, padding=12)
        corr_tab = ttk.Frame(notebook, padding=12)
        notebook.add(outer, text="Proses LFPS")
        notebook.add(corr_tab, text="Korelasi Antar Alat")

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
            ("Jarak node spline (detik)", "spline_spacing_seconds", "60"),
            ("Taper maksimum (0-0.5)", "taper_percentage", "0.05"),
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
        ttk.Label(parameters, text="Metode detrend").grid(
            row=check_row, column=2, sticky="w", padx=(0, 6), pady=3
        )
        ttk.Combobox(
            parameters,
            textvariable=self._variable("detrend_method", "demean + linear"),
            values=("none", "demean", "linear", "demean + linear", "spline"),
            state="readonly",
            width=16,
        ).grid(row=check_row, column=3, sticky="ew", pady=3)
        ttk.Checkbutton(
            parameters,
            text="Taper cosine",
            variable=self._variable("taper", True, tk.BooleanVar),
        ).grid(row=check_row + 1, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            parameters,
            text="Bandpass",
            variable=self._variable("bandpass", True, tk.BooleanVar),
        ).grid(row=check_row + 1, column=1, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            parameters,
            text="Zero-phase bandpass",
            variable=self._variable("zero_phase", True, tk.BooleanVar),
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
        self._build_correlation_tab(corr_tab)

    def _corr_variable(self, name: str, value, cls=tk.StringVar):
        return self._variable("corr_" + name, value, cls)

    def _build_correlation_tab(self, parent) -> None:
        info = ttk.Label(
            parent,
            text=("Bandingkan rekaman beberapa alat yang mengukur di lokasi dan waktu yang sama. "
                  "Output mencakup korelasi Z/N/E, lag, RMS, heatmap, PSD, V/H, dan faktor normalisasi."),
            wraplength=850,
        )
        info.pack(fill="x", pady=(0, 8))
        paths = ttk.LabelFrame(parent, text="Input dan output korelasi", padding=10)
        paths.pack(fill="x")
        for row, (label, name, command) in enumerate((
            ("Folder MiniSEED", "input_folder", self._choose_corr_input),
            ("Folder output", "output_folder", self._choose_corr_output),
        )):
            ttk.Label(paths, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            ttk.Entry(paths, textvariable=self._corr_variable(name, "")).grid(row=row, column=1, sticky="ew")
            ttk.Button(paths, text="Pilih...", command=command).grid(row=row, column=2, padx=(8, 0))
        ttk.Checkbutton(paths, text="Cari di subfolder",
                        variable=self._corr_variable("recursive", False, tk.BooleanVar)).grid(
                            row=2, column=1, sticky="w")
        paths.columnconfigure(1, weight=1)

        cut = ttk.LabelFrame(parent, text="Waktu pengukuran bersama (UTC)", padding=10)
        cut.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(cut, text="Gunakan cut waktu", variable=self._corr_variable(
            "cut_enabled", True, tk.BooleanVar), command=self._toggle_corr_cut).grid(row=0, column=0, sticky="w")
        ttk.Label(cut, text="Starttime").grid(row=1, column=0, sticky="w")
        self.corr_start_entry = ttk.Entry(cut, textvariable=self._corr_variable("cut_start", ""))
        self.corr_start_entry.grid(row=1, column=1, sticky="ew", padx=(6, 14))
        ttk.Label(cut, text="Endtime").grid(row=1, column=2, sticky="w")
        self.corr_end_entry = ttk.Entry(cut, textvariable=self._corr_variable("cut_end", ""))
        self.corr_end_entry.grid(row=1, column=3, sticky="ew", padx=(6, 0))
        cut.columnconfigure(1, weight=1); cut.columnconfigure(3, weight=1)

        settings = ttk.LabelFrame(parent, text="Parameter perbandingan", padding=10)
        settings.pack(fill="x", pady=(8, 0))
        fields = [
            ("Nama alat referensi (opsional)", "reference_name", ""),
            ("Minimum overlap (detik)", "min_overlap", "30"),
            ("Max lag (detik)", "max_lag", "30"),
            ("Window LFPS (detik)", "window_length", "60"),
            ("Overlap LFPS (detik)", "overlap", "30"),
            ("Analisis fmin (Hz)", "analysis_fmin", "1"),
            ("Analisis fmax (Hz)", "analysis_fmax", "6"),
            ("Plot fmin (Hz)", "plot_fmin", "0.5"),
            ("Plot fmax (Hz)", "plot_fmax", "10"),
            ("CFT minimum", "cft_min", "0.5"), ("CFT maksimum", "cft_max", "2"),
            ("Ambang dip", "dip_threshold", "0"),
            ("Bandpass min (Hz)", "preprocess_fmin", "1"),
            ("Bandpass max (Hz)", "preprocess_fmax", "10"),
            ("Order spline", "spline_order", "3"),
            ("Jarak node spline (detik)", "spline_spacing_seconds", "60"),
            ("Taper maksimum (0-0.5)", "taper_percentage", "0.05"),
        ]
        for index, (label, name, default) in enumerate(fields):
            row, pair = divmod(index, 2); column = pair * 2
            ttk.Label(settings, text=label).grid(row=row, column=column, sticky="w", padx=(0, 6), pady=2)
            ttk.Entry(settings, textvariable=self._corr_variable(name, default), width=16).grid(
                row=row, column=column + 1, sticky="ew", padx=(0, 14), pady=2)
        option_row = (len(fields) + 1) // 2
        ttk.Label(settings, text="Window spektral").grid(row=option_row, column=0, sticky="w")
        ttk.Combobox(settings, textvariable=self._corr_variable("spectral_window", "boxcar"),
                     values=("boxcar", "hann", "hamming", "blackman"), state="readonly").grid(
                         row=option_row, column=1, sticky="ew", padx=(0, 14))
        ttk.Label(settings, text="Metode detrend").grid(row=option_row, column=2, sticky="w")
        ttk.Combobox(
            settings, textvariable=self._corr_variable("detrend_method", "demean + linear"),
            values=("none", "demean", "linear", "demean + linear", "spline"), state="readonly"
        ).grid(row=option_row, column=3, sticky="ew")
        ttk.Checkbutton(settings, text="Taper cosine", variable=self._corr_variable(
            "taper", True, tk.BooleanVar)).grid(row=option_row + 1, column=0, sticky="w")
        ttk.Checkbutton(settings, text="Bandpass", variable=self._corr_variable(
            "bandpass", True, tk.BooleanVar)).grid(row=option_row + 1, column=1, sticky="w")
        ttk.Checkbutton(settings, text="Zero-phase bandpass", variable=self._corr_variable(
            "zero_phase", True, tk.BooleanVar)).grid(row=option_row + 1, column=2,
                                                       columnspan=2, sticky="w")
        settings.columnconfigure(1, weight=1); settings.columnconfigure(3, weight=1)

        controls = ttk.Frame(parent); controls.pack(fill="x", pady=(8, 0))
        self.corr_start_button = ttk.Button(controls, text="Jalankan Korelasi", command=self._start_correlation)
        self.corr_start_button.pack(side="left")
        self.corr_cancel_button = ttk.Button(controls, text="Batalkan", command=self._cancel_correlation,
                                             state="disabled")
        self.corr_cancel_button.pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Buka folder output", command=self._open_corr_output).pack(side="left", padx=(8, 0))
        self.corr_status_var = tk.StringVar(value="Siap")
        ttk.Label(controls, textvariable=self.corr_status_var).pack(side="right")
        self.corr_progress = ttk.Progressbar(parent, mode="determinate")
        self.corr_progress.pack(fill="x", pady=(8, 0))
        log_frame = ttk.LabelFrame(parent, text="Log korelasi", padding=6)
        log_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.corr_log = tk.Text(log_frame, height=8, wrap="word", state="disabled")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.corr_log.yview)
        self.corr_log.configure(yscrollcommand=scroll.set)
        self.corr_log.pack(side="left", fill="both", expand=True); scroll.pack(side="right", fill="y")
        self._toggle_corr_cut()

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
            detrend_method=str(self.variables["detrend_method"].get()).lower().replace(" + ", "_"),
            bandpass=bool(self.variables["bandpass"].get()),
            spline_order=int(number("spline_order")),
            spline_spacing_seconds=number("spline_spacing_seconds"),
            taper=bool(self.variables["taper"].get()),
            taper_percentage=number("taper_percentage"),
            zero_phase=bool(self.variables["zero_phase"].get()),
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
        if self.worker is not None or self.corr_worker is not None:
            messagebox.showwarning("Proses aktif", "Tunggu proses lain selesai.", parent=self)
            return
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

    def _choose_corr_input(self) -> None:
        selected = filedialog.askdirectory(title="Pilih folder MiniSEED beberapa alat")
        if selected:
            self.variables["corr_input_folder"].set(selected)
            if not self.variables["corr_output_folder"].get():
                self.variables["corr_output_folder"].set(str(Path(selected) / "LFPS_Output"))
            try:
                files = find_mseed_files(Path(selected), bool(self.variables["corr_recursive"].get()))
                starts, ends = [], []
                for source in files:
                    stream = select_zne(read(str(source), headonly=True))
                    starts.append(max(trace.stats.starttime for trace in stream))
                    ends.append(min(trace.stats.endtime for trace in stream))
                if starts and ends and max(starts) < min(ends):
                    self.variables["corr_cut_start"].set(str(max(starts)))
                    self.variables["corr_cut_end"].set(str(min(ends)))
                self._append_corr_log(f"Ditemukan {len(files)} file; waktu irisan bersama telah diisi.")
            except Exception as exc:
                self._append_corr_log(f"Waktu irisan otomatis tidak dapat dibaca: {exc}")

    def _choose_corr_output(self) -> None:
        selected = filedialog.askdirectory(title="Pilih folder output korelasi")
        if selected:
            self.variables["corr_output_folder"].set(selected)

    def _toggle_corr_cut(self) -> None:
        state = "normal" if self.variables["corr_cut_enabled"].get() else "disabled"
        self.corr_start_entry.configure(state=state); self.corr_end_entry.configure(state=state)

    def _correlation_parameters(self) -> CorrelationParameters:
        def text(name: str) -> str:
            return str(self.variables["corr_" + name].get()).strip()
        def number(name: str) -> float:
            try:
                return float(text(name))
            except ValueError as exc:
                raise ValueError(f"Nilai {name} bukan angka valid.") from exc
        if not text("input_folder") or not text("output_folder"):
            raise ValueError("Folder input dan output korelasi harus dipilih.")
        return CorrelationParameters(
            input_folder=Path(text("input_folder")).expanduser().resolve(),
            output_folder=Path(text("output_folder")).expanduser().resolve(),
            recursive=bool(self.variables["corr_recursive"].get()),
            cut_enabled=bool(self.variables["corr_cut_enabled"].get()),
            cut_start=text("cut_start"), cut_end=text("cut_end"),
            reference_name=text("reference_name"), min_overlap=number("min_overlap"),
            max_lag=number("max_lag"),
            detrend_method=text("detrend_method").lower().replace(" + ", "_"),
            bandpass=bool(self.variables["corr_bandpass"].get()),
            spline_order=int(number("spline_order")),
            spline_spacing_seconds=number("spline_spacing_seconds"),
            taper=bool(self.variables["corr_taper"].get()),
            taper_percentage=number("taper_percentage"),
            zero_phase=bool(self.variables["corr_zero_phase"].get()),
            preprocess_fmin=number("preprocess_fmin"), preprocess_fmax=number("preprocess_fmax"),
            window_length=number("window_length"), overlap=number("overlap"),
            analysis_fmin=number("analysis_fmin"), analysis_fmax=number("analysis_fmax"),
            plot_fmin=number("plot_fmin"), plot_fmax=number("plot_fmax"),
            cft_min=number("cft_min"), cft_max=number("cft_max"),
            dip_threshold=number("dip_threshold"), spectral_window=text("spectral_window").lower())

    def _start_correlation(self) -> None:
        if self.worker is not None or self.corr_worker is not None:
            messagebox.showwarning("Proses aktif", "Tunggu proses lain selesai.", parent=self); return
        try:
            params = self._correlation_parameters(); params.validate()
        except Exception as exc:
            messagebox.showerror("Parameter tidak valid", str(exc), parent=self); return
        self.corr_cancel_event.clear(); self.corr_progress.configure(value=0, maximum=1)
        self.corr_start_button.configure(state="disabled"); self.corr_cancel_button.configure(state="normal")
        self.corr_status_var.set("Memproses...")
        self._append_corr_log(f"Input: {params.input_folder}")
        self.corr_worker = threading.Thread(target=self._run_corr_worker, args=(params,), daemon=True)
        self.corr_worker.start()

    def _run_corr_worker(self, params: CorrelationParameters) -> None:
        try:
            result = process_correlation(params, lambda kind, value: self.events.put((kind, value)),
                                         self.corr_cancel_event.is_set)
            self.events.put(("corr_done", result))
        except Exception as exc:
            self.events.put(("corr_fatal", (exc, traceback.format_exc())))

    def _cancel_correlation(self) -> None:
        self.corr_cancel_event.set(); self.corr_cancel_button.configure(state="disabled")
        self.corr_status_var.set("Membatalkan setelah tahap aktif selesai...")

    def _append_corr_log(self, message: str) -> None:
        self.corr_log.configure(state="normal"); self.corr_log.insert("end", message + "\n")
        self.corr_log.see("end"); self.corr_log.configure(state="disabled")

    def _open_corr_output(self) -> None:
        raw = str(self.variables["corr_output_folder"].get()).strip()
        folder = Path(raw) if raw else None
        if folder is None or not folder.exists():
            messagebox.showwarning("Folder output", "Folder output belum tersedia.", parent=self); return
        os.startfile(folder)

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
                elif kind == "corr_log":
                    self._append_corr_log(str(value))
                elif kind == "corr_total":
                    self.corr_progress.configure(maximum=int(value), value=0)
                elif kind == "corr_progress":
                    self.corr_progress.configure(value=int(value))
                elif kind == "corr_done":
                    path, rows, errors = value
                    self.corr_start_button.configure(state="normal")
                    self.corr_cancel_button.configure(state="disabled")
                    self.corr_worker = None
                    self.corr_status_var.set(f"Selesai: {rows} hasil, {errors} error")
                    self._append_corr_log(f"Ringkasan disimpan: {path}")
                    messagebox.showinfo("Korelasi selesai",
                                        f"Hasil komponen/pasangan: {rows}\nError: {errors}\n\n{path}", parent=self)
                elif kind == "corr_fatal":
                    exc, trace = value
                    self.corr_start_button.configure(state="normal")
                    self.corr_cancel_button.configure(state="disabled")
                    self.corr_worker = None; self.corr_status_var.set("Gagal")
                    self._append_corr_log(trace)
                    messagebox.showerror("Korelasi gagal", str(exc), parent=self)
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
