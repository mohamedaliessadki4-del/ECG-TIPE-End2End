import numpy as np
import pywt


def wavelet_denoise(x: np.ndarray, wavelet: str = "sym4", mode: str = "soft") -> np.ndarray:
    """
    Wavelet denoising using universal threshold.
    sigma estimated using MAD on last detail coefficients.
    """
    x = np.asarray(x).astype(float)

    w = pywt.Wavelet(wavelet)
    maxlev = pywt.dwt_max_level(len(x), w.dec_len)
    coeffs = pywt.wavedec(x, wavelet, level=maxlev)

    detail_last = coeffs[-1]
    sigma = np.median(np.abs(detail_last)) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(x)))

    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        new_coeffs.append(pywt.threshold(c, thr, mode=mode))

    y = pywt.waverec(new_coeffs, wavelet)
    return y[: len(x)]
