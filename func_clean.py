from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from obspy.signal.trigger import classic_sta_lta
from scipy.signal import savgol_filter as sgolay


def _sta_lta(traces, sample_counts):
    """Hitung STA/LTA tiga komponen menggunakan panjang STA asli."""
    return tuple(classic_sta_lta(trace.data, count, count * 3)
                 for trace, count in zip(traces, sample_counts))


def _reject_window(cfts, start, end, minimum, maximum, skip=None):
    """Batas STA/LTA inklusif; skip menandai sampel yang harus diabaikan."""
    return (any(max(cft[start:end]) > maximum or min(cft[start:end]) < minimum
                for cft in cfts)
            or (skip is not None and skip[start:end].any()))


def _smooth_savgol(data, preferred_window=201, polyorder=2):
    """Haluskan data dengan window ganjil yang tidak melebihi panjang data."""
    data = np.asarray(data)
    window = min(preferred_window, len(data))
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return data.copy()
    return sgolay(data, window_length=window, polyorder=polyorder)


def _spectral_window(size, window_type="boxcar"):
    """Buat window spektral: boxcar, Hann, Hamming, atau Blackman."""
    name = str(window_type).strip().lower()
    aliases = {"rectangular": "boxcar", "rectangle": "boxcar", "none": "boxcar"}
    name = aliases.get(name, name)
    builders = {
        "boxcar": np.ones,
        "hann": np.hanning,
        "hamming": np.hamming,
        "blackman": np.blackman,
    }
    if name not in builders:
        choices = ", ".join(builders)
        raise ValueError(f"Window spektral '{window_type}' tidak dikenal. Pilih: {choices}.")
    return builders[name](size)


def _power_spectrum(data, sampling_rate, window_type="boxcar"):
    """Hitung spektrum daya dengan normalisasi energi window."""
    data = np.asarray(data)
    window = _spectral_window(len(data), window_type)
    normalization = sampling_rate * np.sum(window ** 2)
    if normalization <= 0:
        raise ValueError("Energi window spektral harus lebih besar dari nol.")
    return np.abs(np.fft.fft(data * window)) ** 2 / normalization


def _polarization(z, n, e, mirror=False):
    """Rectilinearity, dip, azimuth, eigenvalue; urutan matriks E/N/Z."""
    size = len(z)
    czz, cnn, cee = (np.sum(data * data) / size for data in (z, n, e))
    czn = np.sum(z * n) / size
    cze = np.sum(z * e) / size
    cne = np.sum(n * e) / size
    covariance = np.array([[cee, cne, cze], [cne, cnn, czn], [cze, czn, czz]])
    values, vectors = np.linalg.eig(covariance)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    vector = vectors[:, 0]
    rectilinearity = 1 - (values[1] + values[2]) / (2 * values[0])
    dip = abs(np.rad2deg(np.arctan(vector[2] / np.linalg.norm(vector[:2]))))
    azimuth = np.rad2deg(np.arctan2(vector[1], vector[0]))
    if mirror:
        azimuth = convert_mirror_angle(azimuth)
    return rectilinearity, dip, azimuth, values[0]


def _combine_components(paths, destination):
    """Gabungkan gambar Z/N/E secara vertikal dan tutup semua berkas."""
    from contextlib import ExitStack
    from PIL import Image

    with ExitStack() as stack:
        images = [stack.enter_context(Image.open(path)) for path in paths]
        width, height = images[0].size
        with Image.new('RGB', (width, height * len(images))) as combined:
            for index, image in enumerate(images):
                combined.paste(image, (0, height * index))
            combined.save(destination)


def convert_mirror_angle(math_degree):
    math_degree=math_degree%360
    bearing=(90-math_degree)%360
    if bearing>180:
        bearing-=360
    return bearing

def preprocess(tr, detrend=True, bandpass=True, order=3, dspline=500, freqmin=0.5, freqmax=10):
    for i in range(len(tr)):
        tr[i].data = tr[i].data.astype('float64')
        if detrend:
            tr[i].detrend("spline", order=order, dspline=dspline)
    if bandpass:
        tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax)
    return tr


def calculateVH_STALTA_dip(trdat, windows, shift, savename, fmin=1, fmax=4, fmin_plot=0.5, fmax_plot=10, cft_max=1.5, cft_min=0.5, dip_tres=45, spectral_window="boxcar"):
    if windows < shift:
        print('Ensure windows > shift!')
        return
    tr=trdat.copy()
    cftZ, cftN, cftE = _sta_lta((tr.select(component='Z')[0], tr.select(component='N')[0], tr.select(component='E')[0]), (windows * tr.select(component='Z')[0].stats['sampling_rate'], windows * tr.select(component='N')[0].stats['sampling_rate'], windows * tr.select(component='E')[0].stats['sampling_rate']))

    iskip_sum = np.logical_and.reduce([np.logical_and(cftZ > 0.9999, cftZ < 1.0001), np.logical_and(tr.select(component='Z')[0].data > -0.00000001, tr.select(component='Z')[0].data < 0.00000001)])

    fig_seis = plt.figure()
    ax_seis = fig_seis.add_subplot(2, 1, 1)
    ax_seis.plot(tr.select(component='Z')[0].times("matplotlib"), tr.select(component='Z')[0].data, 'k')
    ax_seis.xaxis_date()
    ax_seis = fig_seis.add_subplot(2, 1, 2)
    ax_seis.plot(tr.select(component='Z')[0].times("matplotlib"), cftZ, 'k')
    ax_seis.xaxis_date()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    iseg = 0
    iseg_count = 0

    t_start = max(tr[0].stats['starttime'], tr[1].stats['starttime'], tr[2].stats['starttime'])
    t_end = min(tr[0].stats['endtime'], tr[1].stats['endtime'], tr[2].stats['endtime'])

    t1 = t_start + iseg * timedelta(seconds=shift)
    t2 = t1 + timedelta(seconds=windows)

    fftZ_list = []
    fftN_list = []
    fftE_list = []
    VperH_list = []

    fig_window = plt.figure()
    ax_winZ = fig_window.add_subplot(3, 1, 1)
    ax_winN = fig_window.add_subplot(3, 1, 2)
    ax_winE = fig_window.add_subplot(3, 1, 3)
    color_iter = plt.cm.rainbow(np.linspace(0, 1, int((t_end - t_start) / shift)))

    while t1 < t_end - windows:
        cft_i_start = int((t1 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])
        cft_i_end = int((t2 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])

        if (_reject_window((cftZ, cftN, cftE), cft_i_start, cft_i_end, cft_min, cft_max, skip=iskip_sum)):
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        ax_seis.axvspan(t1, t2, color=color_iter[iseg], alpha=0.3)

        trc = tr.copy()
        trc.trim(t1, t2)

        #calculate dip
        N = len(trc[0].data)
        L_i, dip_i, azimuth_i, eig_i = _polarization(
            trc.select(component='Z')[0].data, trc.select(component='N')[0].data, trc.select(component='E')[0].data, mirror=False)

        if dip_i<dip_tres:
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        ax_winZ.plot(np.array(trc.select(component='Z')[0].data), alpha=0.3)
        ax_winN.plot(np.array(trc.select(component='N')[0].data), alpha=0.3)
        ax_winE.plot(np.array(trc.select(component='E')[0].data), alpha=0.3)

        samplingrate = tr[0].stats['sampling_rate']
        N = len(trc[0].data)
        fftZ = _power_spectrum(trc.select(component='Z')[0].data, samplingrate, spectral_window)
        fftN = _power_spectrum(trc.select(component='N')[0].data, samplingrate, spectral_window)
        fftE = _power_spectrum(trc.select(component='E')[0].data, samplingrate, spectral_window)

        VperH = np.sqrt((fftZ ** 2) / (fftN ** 2 + fftE ** 2))
        VperH = _smooth_savgol(VperH)

        
        freq = np.fft.fftfreq(N, 1 / samplingrate)
        VperH = VperH[:N // 2]
        freq = freq[:N // 2]

        ax.plot(freq, VperH, alpha=0.1, color=color_iter[iseg])
        fftZ_list.append(fftZ)
        fftN_list.append(fftN)
        fftE_list.append(fftE)
        VperH_list.append(VperH)

        iseg += 1
        iseg_count += 1
        t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
        t2 = t1 + timedelta(seconds=windows)
        trc.clear()

    fig_seis.savefig(savename + '_seis.png')
    plt.close(fig_seis)
    fig_window.savefig(savename + '_windowseis.png')
    plt.close(fig_window)

    if iseg_count == 0:
        ax.set_xlim(fmin_plot, fmax_plot)
        fig.savefig(savename + '.png')
        plt.close(fig)
        return np.nan, np.nan

    VperH_array = np.array(VperH_list)
    Vplot = np.median(VperH_array, axis=0)
    Vmean = np.mean(VperH_array, axis=0)
    freq = np.fft.fftfreq(N, 1 / samplingrate)
    freq = freq[:N // 2]

    q1plot = np.quantile(VperH_array, 0.25, axis=0)
    q3plot = np.quantile(VperH_array, 0.75, axis=0)

    ax.plot(freq, Vplot, color='red', linewidth=2)
    ax.plot(freq, q1plot, color='black', linewidth=1, linestyle='dashed')
    ax.plot(freq, q3plot, color='black', linewidth=1, linestyle='dashed')
    ax.plot(freq, Vmean, color='black', linewidth=2)
    ax.set_ylim(0, 5)
    ax.set_xlim(fmin_plot, fmax_plot)


    finterp = np.arange(fmin, fmax, 0.05)
    vinterp = np.interp(finterp, freq, Vplot)
    finterp_forplot=np.arange(fmin_plot,fmax_plot,0.05)
    vinterp_forplot=np.interp(finterp_forplot,freq,Vplot)
    ax.vlines(x=finterp[vinterp.argmax()],ymin=0,ymax=max(vinterp),colors='red',linestyles='dashdot',label='maximum v/h')
    ax.legend(loc='upper right')
    ax.text(x=finterp[vinterp.argmax()],y=max(vinterp),s="f: "+str(round(finterp[vinterp.argmax()],5))+"\nv/h: "+str(round(max(vinterp),5)),bbox=dict(facecolor='blue', alpha=0.5),wrap=True)
    fig.savefig(savename + '_vhsr.png')
    plt.close(fig)
    tr.clear()
    return max(vinterp), finterp[vinterp.argmax()], finterp, vinterp, finterp_forplot, vinterp_forplot

def _calculate_psd_dip(trdat, windows, shift, savename, fmin=1, fmax=4, fmin_plot=0.5, fmax_plot=10, cft_max=1.5, cft_min=0.5, dip_tres=45, *, logarithmic=True, spectral_window="boxcar"):
    if windows < shift:
        print('Ensure windows > shift!')
        return
    suffix = 'log' if logarithmic else ''
    tr=trdat.copy()
    cftZ, cftN, cftE = _sta_lta((tr.select(component='Z')[0], tr.select(component='N')[0], tr.select(component='E')[0]), (windows * tr.select(component='Z')[0].stats['sampling_rate'], windows * tr.select(component='N')[0].stats['sampling_rate'], windows * tr.select(component='E')[0].stats['sampling_rate']))

    iskip_sum = np.logical_and.reduce([np.logical_and(cftZ > 0.9999, cftZ < 1.0001), np.logical_and(tr.select(component='Z')[0].data > -0.00000001, tr.select(component='Z')[0].data < 0.00000001)])

    fig_seis = plt.figure()
    ax_seis = fig_seis.add_subplot(2, 1, 1)
    ax_seis.plot(tr.select(component='Z')[0].times("matplotlib"), tr.select(component='Z')[0].data, 'k')
    ax_seis.xaxis_date()
    ax_seis = fig_seis.add_subplot(2, 1, 2)
    ax_seis.plot(tr.select(component='Z')[0].times("matplotlib"), cftZ, 'k')
    ax_seis.xaxis_date()

    figZ, axZ = plt.subplots()
    figN, axN = plt.subplots()
    figE, axE = plt.subplots()

    iseg = 0
    t_start = max(tr[0].stats['starttime'], tr[1].stats['starttime'], tr[2].stats['starttime'])
    t_end = min(tr[0].stats['endtime'], tr[1].stats['endtime'], tr[2].stats['endtime'])
    t1 = t_start + iseg * timedelta(seconds=shift)
    t2 = t1 + timedelta(seconds=windows)

    fftZ_sum = np.zeros(int(((windows * tr.select(component='Z')[0].stats['sampling_rate']) + 1) // 2))
    fftN_sum = np.zeros_like(fftZ_sum)
    fftE_sum = np.zeros_like(fftZ_sum)

    fftZ_list=[]
    fftN_list=[]
    fftE_list=[]

    iseg_count = 0
    color_iter = plt.cm.rainbow(np.linspace(0, 1, int((t_end - t_start) / shift)))

    while t1 < t_end - windows:
        cft_i_start = int((t1 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])
        cft_i_end = int((t2 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])

        if (_reject_window((cftZ, cftN, cftE), cft_i_start, cft_i_end, cft_min, cft_max, skip=iskip_sum)):
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        ax_seis.axvspan(t1, t2, color=color_iter[iseg], alpha=0.3)
        trc = tr.copy()
        trc.trim(t1, t2)

        #calculate dip
        N = len(trc[0].data)
        L_i, dip_i, azimuth_i, eig_i = _polarization(
            trc.select(component='Z')[0].data, trc.select(component='N')[0].data, trc.select(component='E')[0].data, mirror=False)

        if dip_i<dip_tres:
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        samplingrate = tr[0].stats['sampling_rate']
        N = len(trc[0].data)
        fftZ = _power_spectrum(trc.select(component='Z')[0].data, samplingrate, spectral_window)
        fftN = _power_spectrum(trc.select(component='N')[0].data, samplingrate, spectral_window)
        fftE = _power_spectrum(trc.select(component='E')[0].data, samplingrate, spectral_window)
        fftZ, fftN, fftE = fftZ[:N // 2], fftN[:N // 2], fftE[:N // 2]

        fftZlog = _smooth_savgol(10 * np.log10(fftZ) if logarithmic else fftZ)
        fftNlog = _smooth_savgol(10 * np.log10(fftN) if logarithmic else fftN)
        fftElog = _smooth_savgol(10 * np.log10(fftE) if logarithmic else fftE)

        freq = np.fft.fftfreq(N, 1 / samplingrate)[:N // 2]

        axZ.plot(freq, fftZlog, alpha=0.1, color=color_iter[iseg])
        axN.plot(freq, fftNlog, alpha=0.1, color=color_iter[iseg])
        axE.plot(freq, fftElog, alpha=0.1, color=color_iter[iseg])

        fftZ_sum += fftZ
        fftN_sum += fftN
        fftE_sum += fftE
        fftZ_list.append(fftZlog)
        fftN_list.append(fftNlog)
        fftE_list.append(fftElog)

        iseg += 1
        iseg_count += 1
        t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
        t2 = t1 + timedelta(seconds=windows)
        trc.clear()

    fig_seis.savefig(savename + '_seis.png')
    plt.close(fig_seis)

    if iseg_count == 0:
        for ax in [axZ, axN, axE]:
            ax.set_xlim(fmin_plot, fmax_plot)
            ax.set_ylim(0, 150)
        figZ.savefig(savename + '_Z.png')
        figN.savefig(savename + '_N.png')
        figE.savefig(savename + '_E.png')
        plt.close(figZ)
        plt.close(figN)
        plt.close(figE)
        return np.nan, np.nan, np.nan


    fftZ_list = np.array(fftZ_list)
    fftN_list = np.array(fftN_list)
    fftE_list = np.array(fftE_list)
    fftZmed = np.median(fftZ_list, axis=0)
    fftZmean = np.mean(fftZ_list, axis=0)
    fftNmed = np.median(fftN_list, axis=0)
    fftNmean = np.mean(fftN_list, axis=0)
    fftEmed = np.median(fftE_list, axis=0)
    fftEmean = np.mean(fftE_list, axis=0)
    q1Z = np.quantile(fftZ_list, 0.25, axis=0)
    q3Z = np.quantile(fftZ_list, 0.75, axis=0)
    q1N = np.quantile(fftN_list, 0.25, axis=0)
    q3N = np.quantile(fftN_list, 0.75, axis=0)
    q1E = np.quantile(fftE_list, 0.25, axis=0)
    q3E = np.quantile(fftE_list, 0.75, axis=0)

    freq = np.fft.fftfreq(N, 1 / samplingrate)[:N // 2]

    axZ.plot(freq, fftZmean, color='black', linewidth=2)
    axZ.plot(freq, q1Z, color='black', linewidth=1, linestyle='dashed')
    axZ.plot(freq, q3Z, color='black', linewidth=1, linestyle='dashed')
    axZ.plot(freq, fftZmed, color='red', linewidth=2)
    axZ.set_xlim(fmin_plot, fmax_plot)

    axN.plot(freq, fftNmean, color='black', linewidth=2)
    axN.plot(freq, q1N, color='black', linewidth=1, linestyle='dashed')
    axN.plot(freq, q3N, color='black', linewidth=1, linestyle='dashed')
    axN.plot(freq, fftNmed, color='red', linewidth=2)
    axN.set_xlim(fmin_plot, fmax_plot)

    axE.plot(freq, fftEmean, color='black', linewidth=2)
    axE.plot(freq, q1E, color='black', linewidth=1, linestyle='dashed')
    axE.plot(freq, q3E, color='black', linewidth=1, linestyle='dashed')
    axE.plot(freq, fftEmed, color='red', linewidth=2)
    axE.set_xlim(fmin_plot, fmax_plot)

    finterp = np.arange(fmin, fmax, 0.05)
    Zinterp = np.interp(finterp, freq, fftZmean)
    finterp_forplot=np.arange(fmin_plot,fmax_plot,0.05)
    Zinterp_forplot=np.interp(finterp_forplot,freq,fftZmean)

    figZ.savefig(savename + '_Z' + suffix + '.png')
    figN.savefig(savename + '_N' + suffix + '.png')
    figE.savefig(savename + '_E' + suffix + '.png')

    #integral
    verts = [(fmin, 0), *zip(finterp[(finterp >= fmin) & (finterp <= fmax)], Zinterp[(finterp >= fmin) & (finterp <= fmax)]), (fmax, 0)]
    poly = Polygon(verts, facecolor='0.5', edgecolor='0.5')
    psd_iz = np.trapz(Zinterp[(finterp >= fmin) & (finterp <= fmax)], x=finterp[(finterp >= fmin) & (finterp <= fmax)])
    axZ.add_patch(poly)

    axZ.vlines(x=finterp[Zinterp.argmax()],ymin=0,ymax=max(Zinterp),colors='red',linestyles='dashdot',label='maximum PSD-Z')
    axZ.legend(loc='upper right')
    axZ.text(x=finterp[Zinterp.argmax()], y=max(Zinterp),
            s="f: " + str(round(finterp[Zinterp.argmax()],5)) + "\nZmax: " + str(round(max(Zinterp),5))+ "\nIZ: "+ str(round(psd_iz,5)),
            bbox=dict(facecolor='blue', alpha=0.5), wrap=True)

    figZ.savefig(savename + '_Zintegral' + suffix + '.png')
    plt.close(figZ)
    plt.close(figN)
    plt.close(figE)

    _combine_components((savename + '_Z' + suffix + '.png', savename + '_N' + suffix + '.png', savename + '_E' + suffix + '.png'), savename + '_ZNE' + suffix + '.png')
    tr.clear()
    return max(Zinterp), finterp[Zinterp.argmax()], psd_iz, finterp, Zinterp, finterp_forplot, Zinterp_forplot

def calPSD_STALTA_dip(trdat, windows, shift, savename, fmin=1, fmax=4, fmin_plot=0.5, fmax_plot=10, cft_max=1.5, cft_min=0.5, dip_tres=45, spectral_window="boxcar"):
    """PSD dengan seleksi dip; skala dB."""
    return _calculate_psd_dip(
        trdat, windows, shift, savename, fmin, fmax, fmin_plot, fmax_plot,
        cft_max, cft_min, dip_tres, logarithmic=True, spectral_window=spectral_window)

def calPSD_STALTA_dip_nolog(trdat, windows, shift, savename, fmin=1, fmax=4, fmin_plot=0.5, fmax_plot=10, cft_max=1.5, cft_min=0.5, dip_tres=45, spectral_window="boxcar"):
    """PSD dengan seleksi dip; skala linear."""
    return _calculate_psd_dip(
        trdat, windows, shift, savename, fmin, fmax, fmin_plot, fmax_plot,
        cft_max, cft_min, dip_tres, logarithmic=False, spectral_window=spectral_window)

def cal_polar_dip(trdat, windows, shift, savename, fmin=1, fmax=4, fmin_plot=0.5, fmax_plot=10, cft_max=1.5, cft_min=0.5, dip_tres=45):
    if windows < shift:
        print('Ensure windows > shift!')
        return

    tr = trdat.copy()
    t_start = max(tr[0].stats['starttime'], tr[1].stats['starttime'], tr[2].stats['starttime'])
    t_end = min(tr[0].stats['endtime'], tr[1].stats['endtime'], tr[2].stats['endtime'])
    tr.trim(t_start, t_end)

    cftZ, cftN, cftE = _sta_lta((tr.select(component='Z')[0], tr.select(component='N')[0], tr.select(component='E')[0]), (windows * tr.select(component='Z')[0].stats['sampling_rate'], windows * tr.select(component='N')[0].stats['sampling_rate'], windows * tr.select(component='E')[0].stats['sampling_rate']))

    iskip_sum = np.logical_and.reduce([np.logical_and(cftZ > 0.9999, cftZ < 1.0001), np.logical_and(tr.select(component='Z')[0].data > -0.00000001, tr.select(component='Z')[0].data < 0.00000001)])

    idx_sel = np.logical_and.reduce([
        np.logical_and(cftZ > cft_min, cftZ < cft_max),
        np.logical_and(cftN > cft_min, cftN < cft_max),
        np.logical_and(cftE > cft_min, cftE < cft_max),
        ~iskip_sum
    ])

    dataZ, dataN, dataE = tr.select(component='Z')[0].data[idx_sel], tr.select(component='N')[0].data[idx_sel], tr.select(component='E')[0].data[idx_sel]
    N = len(dataZ)
    L_all, dip_all, azimuth_all, eig_all = _polarization(
        dataZ, dataN, dataE, mirror=True)

    iseg = 0
    t1 = t_start + iseg * timedelta(seconds=shift)
    t2 = t1 + timedelta(seconds=windows)

    L_list, dip_list, azimuth_list, eig_list, t_list, iseg_list = [], [], [], [], [], []

    while t1 < t_end - windows:
        cft_i_start = int((t1 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])
        cft_i_end = int((t2 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])

        if (_reject_window((cftZ, cftN, cftE), cft_i_start, cft_i_end, cft_min, cft_max, skip=iskip_sum)):
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        trc = tr.copy()
        trc.trim(t1, t2)

        N = len(trc[0].data)
        L_i, dip_i, azimuth_i, eig_i = _polarization(
            trc.select(component='Z')[0].data, trc.select(component='N')[0].data, trc.select(component='E')[0].data, mirror=True)
        if dip_i<dip_tres:
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        L_list.append(L_i)
        dip_list.append(dip_i)
        azimuth_list.append(azimuth_i)
        eig_list.append(eig_i)
        t_list.append(iseg * shift + windows / 2)
        iseg_list.append(iseg)

        iseg += 1
        t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
        t2 = t1 + timedelta(seconds=windows)
        trc.clear()

    min_val = min(iseg_list)
    max_val = max(iseg_list)
    normalized_values = [(val - min_val) / (max_val - min_val) for val in iseg_list]
    color_list = [plt.cm.rainbow(val) for val in normalized_values]

    fig = plt.figure(figsize=(7, 5))
    ax_L = fig.add_subplot(2, 2, 1)
    ax_L.scatter(t_list, L_list, c=color_list)
    ax_L.axhline(L_all, color='black')
    ax_L.axhline(np.mean(L_list), color='red', linestyle='dashed')
    ax_L.axhline(np.quantile(L_list,0.25), color='black', linestyle='dashed')
    ax_L.axhline(np.quantile(L_list,0.75), color='black', linestyle='dashed')
    ax_L.set_ylabel('Rectilinearity')
    ax_L.set_ylim(0, 1)

    ax_dip = fig.add_subplot(2, 2, 2)
    ax_dip.scatter(t_list, dip_list, c=color_list)
    ax_dip.axhline(dip_all, color='black')
    ax_dip.axhline(np.mean(dip_list), color='red', linestyle='dashed')
    ax_dip.axhline(np.quantile(dip_list,0.25), color='black', linestyle='dashed')
    ax_dip.axhline(np.quantile(dip_list,0.75), color='black', linestyle='dashed')
    ax_dip.set_ylabel('Dip (°)')
    ax_dip.set_ylim(0, 90)

    ax_azimuth = fig.add_subplot(2, 2, 3)
    ax_azimuth.scatter(t_list, azimuth_list, c=color_list)
    ax_azimuth.axhline(azimuth_all, color='black')
    ax_azimuth.axhline(np.mean(azimuth_list), color='red', linestyle='dashed')
    ax_azimuth.axhline(np.quantile(azimuth_list,0.25), color='black', linestyle='dashed')
    ax_azimuth.axhline(np.quantile(azimuth_list,0.75), color='black', linestyle='dashed')
    ax_azimuth.set_ylabel('Azimuth (°)')
    ax_azimuth.set_xlabel('(s) from start time')
    ax_azimuth.set_ylim(-180, 180)

    ax_eig = fig.add_subplot(2, 2, 4)
    ax_eig.scatter(t_list, eig_list, c=color_list)
    ax_eig.axhline(eig_all, color='black')
    ax_eig.axhline(np.mean(eig_list), color='red', linestyle='dashed')
    ax_eig.axhline(np.quantile(eig_list,0.25), color='black', linestyle='dashed')
    ax_eig.axhline(np.quantile(eig_list,0.75), color='black', linestyle='dashed')
    ax_eig.set_xlabel('(s) from start time')
    ax_eig.set_ylabel('Largest Eigenvalue')

    fig.savefig(savename + ".png")
    plt.close(fig)
    tr.clear()
    return (L_all, dip_all, azimuth_all, eig_all), (np.mean(L_list),np.mean(dip_list), np.mean(azimuth_list), np.mean(eig_list))

def plot_window_dip(trdat, windows, shift, savename, fmin=1, fmax=4, fmin_plot=0.5, fmax_plot=10, cft_max=1.5, cft_min=0.5,dip_tres=45):
    if windows < shift:
        print('Ensure windows > shift!')
        return

    tr=trdat.copy()
    cftZ, cftN, cftE = _sta_lta((tr.select(component='Z')[0], tr.select(component='N')[0], tr.select(component='E')[0]), (windows * tr.select(component='Z')[0].stats['sampling_rate'], windows * tr.select(component='N')[0].stats['sampling_rate'], windows * tr.select(component='E')[0].stats['sampling_rate']))

    iskip_sum = np.logical_and.reduce([np.logical_and(cftZ > 0.9999, cftZ < 1.0001), np.logical_and(tr.select(component='Z')[0].data > -0.00000001, tr.select(component='Z')[0].data < 0.00000001)])

    fig_seis = plt.figure()
    ax_seisZ = fig_seis.add_subplot(3, 1, 1)
    ax_seisZ.plot(tr.select(component='Z')[0].times("matplotlib"), tr.select(component='Z')[0].data, 'k')
    ax_seisZ.xaxis_date()
    ax_seisN = fig_seis.add_subplot(3, 1, 2)
    ax_seisN.plot(tr.select(component='N')[0].times("matplotlib"), tr.select(component='N')[0].data, 'k')
    ax_seisN.xaxis_date()
    ax_seisE = fig_seis.add_subplot(3, 1, 3)
    ax_seisE.plot(tr.select(component='E')[0].times("matplotlib"), tr.select(component='E')[0].data, 'k')
    ax_seisE.xaxis_date()

    iseg = 0
    t_start = max(tr[0].stats['starttime'], tr[1].stats['starttime'], tr[2].stats['starttime'])
    t_end = min(tr[0].stats['endtime'], tr[1].stats['endtime'], tr[2].stats['endtime'])
    t1 = t_start + iseg * timedelta(seconds=shift)
    t2 = t1 + timedelta(seconds=windows)

    iseg_count = 0
    color_iter = plt.cm.rainbow(np.linspace(0, 1, int((t_end - t_start) / shift)))
    A_datZ=[]
    A_datN=[]
    A_datE=[]
    while t1 < t_end - windows:
        cft_i_start = int((t1 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])
        cft_i_end = int((t2 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])

        if (_reject_window((cftZ, cftN, cftE), cft_i_start, cft_i_end, cft_min, cft_max, skip=iskip_sum)):
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        trc = tr.copy()
        trc.trim(t1, t2)
        #caldip
        N = len(trc[0].data)
        L_i, dip_i, azimuth_i, eig_i = _polarization(
            trc.select(component='Z')[0].data, trc.select(component='N')[0].data, trc.select(component='E')[0].data, mirror=False)
        if dip_i<dip_tres:
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        ax_seisZ.axvspan(t1, t2, color=color_iter[iseg], alpha=0.3)
        ax_seisN.axvspan(t1, t2, color=color_iter[iseg], alpha=0.3)
        ax_seisE.axvspan(t1, t2, color=color_iter[iseg], alpha=0.3)

        iseg += 1
        iseg_count += 1
        t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
        t2 = t1 + timedelta(seconds=windows)
        A_datZ.append(np.array(trc.select(component='Z')[0].data))
        A_datN.append(np.array(trc.select(component='N')[0].data))
        A_datE.append(np.array(trc.select(component='E')[0].data))
        trc.clear()

    fig_seis.savefig(savename + '_selwindow.png')
    plt.close(fig_seis)

    percent = (iseg_count / iseg) * 100
    percent_2h = (iseg_count / 239) * 100
    group = 'good'
    if percent < 60:
        group = 'moderate'
    if percent < 30:
        group = 'poor'
    group_2h = 'good'
    if percent_2h < 60:
        group_2h = 'moderate'
    if percent_2h < 30:
        group_2h = 'poor'
    A_datZ=np.hstack(A_datZ)
    A_datN=np.hstack(A_datN)
    A_datE=np.hstack(A_datE)
    Az_rms=np.sqrt(np.mean(A_datZ**2))
    An_rms=np.sqrt(np.mean(A_datN**2))
    Ae_rms=np.sqrt(np.mean(A_datE**2))
    tr.clear()
    return percent, group, percent_2h, group_2h, Az_rms,An_rms, Ae_rms
