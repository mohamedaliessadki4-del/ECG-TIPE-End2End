import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def compute_fft(signal, fs):
    """
    Compute Fourier spectrum of ECG signal.
    """

    N = len(signal)

    yf = fft(signal)
    xf = fftfreq(N, 1 / fs)

    return xf[:N // 2], np.abs(yf[:N // 2])


def plot_fft(signal, fs):

    freqs, spectrum = compute_fft(signal, fs)

    plt.figure(figsize=(8,4))
    plt.plot(freqs, spectrum)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("ECG Frequency Spectrum")
    plt.grid()
    plt.show()
