import numpy as np
from scipy.signal import convolve, butter, lfilter, freqz, kaiserord, firwin

def sweep(iq_data, fs, freq_band, delta_t, dB, filtered=True, order=20, coherence_time=5):
    f0, f1 = freq_band
    data_len = iq_data.shape[0]

    data_complex = iq_data[:, 0] + 1j * iq_data[:, 1]
    # t = np.arange(0,iq_data.shape[0])/fs

    data_LT = filter_sweep(data_complex, fs, f1, f0)

    energy_of_data = np.sum(np.abs(data_LT) ** 2)

    t0 = 0  # sec
    ts = 1 / fs
    b = (f1 - f0) / (delta_t - t0)

    sweep = np.zeros(data_len, dtype="complex")
    t = np.arange(t0, delta_t, ts)
    t_len = len(t)

    for k in np.arange(0, data_len // t_len):
        del_b = 0.5 * b * np.random.rand()
        if (k % coherence_time == 0):
            rayleigh = np.random.rayleigh(size=1, scale=0.5)[0]
        else:
            rayleigh = 1
        # sweep[t_len*k:t_len*(k+1)]=rayleigh* channel(np.exp(1j*2*np.pi*(f0+0.5*(b+del_b)*t)*t),ts)
        sweep[t_len * k:t_len * (k + 1)] = rayleigh * np.exp(1j * 2 * np.pi * (f0 + 0.5 * (b + del_b) * t) * t)

    if (filtered):
        filt_sw = filter_sweep(sweep, fs, f0, f1)

        energy_of_sweep = np.sum(np.abs(filt_sw) ** 2)

        gain = 10 ** (dB / 10)
        param = gain * energy_of_data / energy_of_sweep

        filt_sw = np.sqrt(param) * filt_sw

        new_complex = data_complex + filt_sw
        iq_new1 = np.zeros(iq_data.shape)
        iq_new1[:, 0] = np.real(new_complex)
        iq_new1[:, 1] = np.imag(new_complex)

        return iq_new1
    else:
        iq_new1 = np.zeros((data_len, 2))
        iq_new1[:, 0] = np.real(sweep)
        iq_new1[:, 1] = np.imag(sweep)

        return iq_new1


def filter_sweep(data, fs, f0, f1, order=20):
    f_co = (abs(f0 - f1)) / 2
    filt = butter_lowpass(f_co, fs, order=order)
    fc = (f0 + f1) / 2
    t = np.arange(0, data.size) / fs
    data *= np.exp(-1j * 2 * np.pi * fc * t)  # move to bb
    data = lfilter(filt[0], filt[1], data)  # filter
    data *= np.exp(1j * 2 * np.pi * fc * t)  # bring back

    return data

def butter_lowpass(cutoff, fs, order=5):
    nyq_rate = fs/2
    normal_cutoff = cutoff / nyq_rate
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def CW(iq_data, fs, fc, dB):
    data_complex = iq_data[:, 0] + 1j * iq_data[:, 1]
    t = np.arange(0, iq_data.shape[0]) / fs

    upper_bound, lower_bound = 1e7, -1e7
    data_LT = filter_sweep(data_complex, fs, upper_bound, lower_bound)
    energy_of_data = np.sum(np.abs(data_LT) ** 2)

    rand_phase = np.random.random()*2*np.pi
    dis_complex = np.exp( 1j*(2*np.pi*fc*t + rand_phase) )
    energy_of_CW = np.sum(np.abs(dis_complex) ** 2)

    gain = 10 ** (dB / 10)
    param = gain * energy_of_data / energy_of_CW

    dis_complex = np.sqrt(param) * dis_complex

    new_energy_of_CW = np.sum(np.abs(dis_complex) ** 2)

    new_complex = data_complex + dis_complex
    iq_new = np.zeros(iq_data.shape)
    iq_new[:, 0] = np.real(new_complex)
    iq_new[:, 1] = np.imag(new_complex)

    return iq_new
