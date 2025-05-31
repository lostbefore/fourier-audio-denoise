import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft, fftfreq

def plot_signal(signal, rate, title, filename):
    t = np.linspace(0, len(signal) / rate, num=len(signal))
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_spectrum(freq, spectrum, title, filename):
    plt.figure(figsize=(10, 4))
    plt.plot(freq, np.abs(spectrum))
    plt.title(title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_compare_signals(original, filtered, rate, filename):
    t = np.linspace(0, len(original) / rate, num=len(original))
    plt.figure(figsize=(10, 4))
    plt.plot(t, original, label="Original", alpha=0.6)
    plt.plot(t, filtered, label="Denoised", alpha=0.8)
    plt.title("Original vs Denoised Signal (Time Domain)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def apply_low_pass_filter(spectrum, freqs, cutoff):
    filtered = np.copy(spectrum)
    filtered[np.abs(freqs) > cutoff] = 0
    return filtered

def main():
   
    rate, data = wavfile.read("input.wav")
    if data.ndim > 1:
        data = data[:, 0]  # 如果是立体声，只处理左声道

    #傅里叶变换
    spectrum = fft(data)
    freqs = fftfreq(len(data), 1 / rate)

   
    plot_signal(data, rate, "Original Signal", "original_signal.png")
    plot_spectrum(freqs, spectrum, "Original Spectrum", "original_spectrum.png")

    #滤波处理
    cutoff = 4000  # 频率阈值（Hz）
    filtered_spectrum = apply_low_pass_filter(spectrum, freqs, cutoff)

    #逆变换为时域信号
    filtered_signal = np.real(ifft(filtered_spectrum))

   
    plot_signal(filtered_signal, rate, "Filtered Signal", "filtered_signal.png")
    plot_spectrum(freqs, filtered_spectrum, "Filtered Spectrum", "filtered_spectrum.png")

   
    plot_compare_signals(data, filtered_signal, rate, "compare_signal.png")

  
    filtered_signal = np.int16(filtered_signal / np.max(np.abs(filtered_signal)) * 32767)
    wavfile.write("output.wav", rate, filtered_signal)

    print("除噪完成，结果保存在 output.wav 和 compare_signal.png")

if __name__ == "__main__":
    main()
