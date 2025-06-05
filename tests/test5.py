import torch

# Create a signal
signal = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
print("Original signal:", signal)

# Fourier Transform
print("\nFourier Transform:")
fft_result = torch.fft.fft(signal)
print("FFT:", fft_result)

# Inverse Fourier Transform
print("\nInverse Fourier Transform:")
ifft_result = torch.fft.ifft(fft_result)
print("IFFT:", ifft_result)

# Real FFT
print("\nReal FFT:")
rfft_result = torch.fft.rfft(signal)
print("RFFT:", rfft_result)

# Short-time Fourier Transform (STFT)
print("\nShort-time Fourier Transform:")
stft_result = torch.stft(signal, n_fft=4, return_complex=True)
print("STFT shape:", stft_result.shape)
print("STFT:", stft_result)

# Window functions
print("\nWindow functions:")
hamming = torch.hamming_window(8)
hann = torch.hann_window(8)
blackman = torch.blackman_window(8)
print("Hamming window:", hamming)
print("Hann window:", hann)
print("Blackman window:", blackman)

# Frequency shifting
print("\nFrequency shifting:")
shifted = torch.fft.fftshift(fft_result)
print("Shifted FFT:", shifted) 