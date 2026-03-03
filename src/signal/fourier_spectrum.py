import numpy as np
import matplotlib.pyplot as plt


def plot_fft_spectrum(signal: np.ndarray, fs: float = 360.0, out_png: str = "spectre_fourier.png"):
    signal = np.asarray(signal).astype(float)
    window = np.hanning(len(signal))
    sig_w = signal * window

    n = len(sig_w)
    fft_spec = np.fft.rfft(sig_w)
    amplitude = np.abs(fft_spec) / (n / 2)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    plt.figure(figsize=(8, 4))
    plt.plot(freqs, amplitude, linewidth=0.6)
    plt.title("ECG Fourier Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()
