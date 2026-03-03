import time
from typing import List
import matplotlib.pyplot as plt


def read_ads1115_samples(
    n_samples: int = 700,
    gain: float = 2 / 3,
    dt: float = 0.002,
    channel: int = 0,
) -> List[int]:
    """
    Read n_samples from ADS1115 (channel A0 by default).
    Returns raw ADC integer values.
    """
    try:
        import board
        import busio
        import adafruit_ads1x15.ads1115 as ADS
        from adafruit_ads1x15.analog_in import AnalogIn

        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS.ADS1115(i2c)
        ads.gain = gain
        chan = AnalogIn(ads, getattr(ADS, f"P{channel}"))

        values = []
        for _ in range(n_samples):
            values.append(chan.value)
            time.sleep(dt)
        return values

    except Exception:
        # fallback for older library
        import Adafruit_ADS1x15  # type: ignore

        adc = Adafruit_ADS1x15.ADS1115()
        values = []
        for _ in range(n_samples):
            values.append(adc.read_adc(channel, gain=gain))
            time.sleep(dt)
        return values


def plot_raw(values: List[int], title: str = "ECG raw (ADS1115)"):
    plt.plot(range(len(values)), values)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("ADC raw value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    vals = read_ads1115_samples()
    plot_raw(vals)
