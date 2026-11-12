# LFPS Processor
LFPS Processor adalah program untuk mengolah data waveform tiga komponen dalam format MiniSEED. Program ini menyediakan analisis LFPS untuk satu file maupun satu folder, serta perbandingan hasil pengukuran dari beberapa alat yang ditempatkan di lokasi dan waktu yang sama.

## Fitur utama
- Pemrosesan satu file `.mseed`/`.miniseed` atau seluruh file dalam folder.
- Pencarian file secara rekursif hingga subfolder.
- Pemotongan data berdasarkan `starttime` dan `endtime` UTC.
- Seleksi window berdasarkan STA/LTA dan dip.
- Perhitungan dan plot:
  - PSD-Z linear dan logaritmik;
  - integral PSD-Z;
  - rasio V/H;
  - polarisasi, dip, azimuth, rectilinearity, dan eigenvalue;
  - RMS komponen Z, N, dan E;
  - kualitas window pengukuran.
- Pilihan window spektral Boxcar, Hann, Hamming, dan Blackman.
- Savitzky–Golay smoothing otomatis berdasarkan resolusi frekuensi dengan lebar default `0.5 Hz`.
- Pilihan preprocessing `none`, `demean`, `linear`, `demean + linear`, dan `spline`.
- Taper cosine dan zero-phase bandpass.
- Tab korelasi untuk membandingkan beberapa alat:
  - korelasi maksimum setiap pasangan alat pada komponen Z/N/E;
  - lag time dan Pearson correlation pada zero lag;
  - RMS difference mentah dan ternormalisasi;
  - heatmap korelasi, lag, dan RMS;
  - perbandingan PSD dan V/H;
  - faktor normalisasi terhadap alat referensi.
- Progress bar, pembatalan proses, log parameter, dan rekap hasil.

## Persyaratan data
Setiap file MiniSEED harus memiliki tiga trace yang dapat dikenali dari karakter terakhir kode channel:
*Z  komponen vertikal
*N  komponen utara–selatan
*E  komponen timur–barat
Contoh channel yang valid adalah `HHZ`, `HHN`, dan `HHE`. Ketiga komponen harus mempunyai rentang waktu yang saling beririsan.

## Instalasi dari source code
-Gunakan Python 3.11 64-bit pada Windows.
-git clone https://github.com/radityaambara/LFPS_Gui.git
-cd LFPS_Gui
-python -m pip install --upgrade pip
-python -m pip install numpy==1.26.4 scipy==1.13.1 matplotlib==3.9.4 pandas==2.2.3 obspy==1.4.2 openpyxl==3.1.5 Pillow==11.3.0 greenlet==3.1.1
-python lfps_gui.py

## Menggunakan tab Proses LFPS
1. Pilih mode **Batch folder** atau **Satu file**.
2. Pilih folder/file MiniSEED dan folder output.
3. Untuk satu file, aktifkan trim waktu jika hanya bagian tertentu yang akan dianalisis.
4. Periksa parameter LFPS dan preprocessing.
5. Klik **Jalankan LFPS**.
6. Pantau progress dan pesan pada panel log.

## Menggunakan tab Korelasi Antar Alat
Tab ini digunakan ketika beberapa alat merekam lokasi yang sama pada waktu yang sama.
1. Tempatkan file MiniSEED seluruh alat dalam satu folder.
2. Pilih folder input dan output.
3. Gunakan rentang cut UTC yang sama untuk seluruh alat.
4. Masukkan nama alat referensi jika diperlukan. Jika kosong, file pertama digunakan sebagai referensi.
5. Tentukan minimum overlap dan maksimum lag.
6. Klik **Jalankan Korelasi**.

Jika sampling rate berbeda, kedua trace otomatis diinterpolasi ke sampling rate terendah sebelum korelasi dihitung.

## Output LFPS
Folder output utama berisi:
"""
LFPS_Output/
├── Hasil_Akhir_LFPS.xlsx
├── LFPS_Run_Log.json
└── <nama_stasiun>/
    ├── plot waveform dan window
    ├── plot PSD-Z
    ├── plot integral PSD-Z
    ├── plot V/H
    └── plot polarisasi
"""
Jika engine Excel tidak tersedia, rekap otomatis disimpan sebagai CSV.

## Output korelasi
Correlation_Comparison/
├── pair/
├── heatmap/
├── psd/
├── vhsr/
├── REKAP_XCORR_FINAL.csv
├── REKAP_NORMALIZATION.csv
└── Correlation_Run_Log.json

##
Masukan dan kontribusi dapat menghubungi di radityaambara@gmail.com
##
