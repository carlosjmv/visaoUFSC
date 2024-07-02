import numpy as np
from scipy import signal

class Signal_processing():
    """
    This class contains various signal processing functions used in the rPPG application.
    """
    def __init__(self):
        self.a = 1

    def normalization(self, data_buffer):
        """
        Normalize the input data buffer.

        Args:
            data_buffer (numpy array): The input data buffer.

        Returns:
            normalized_data (numpy array): The normalized data buffer.
        """
        normalized_data = data_buffer / np.linalg.norm(data_buffer)
        return normalized_data

    def signal_detrending(self, data_buffer):
        """
        Remove the overall trending from the input data buffer.

        Args:
            data_buffer (numpy array): The input data buffer.

        Returns:
            detrended_data (numpy array): The detrended data buffer.
        """
        detrended_data = signal.detrend(data_buffer)
        return detrended_data

    def interpolation(self, data_buffer, times):
        """
        Perform linear interpolation on the data buffer to make the signal more periodic (avoid spectral leakage).

        Args:
            data_buffer (numpy array): The input data buffer.
            times (numpy array): The corresponding time values for the data buffer.

        Returns:
            interpolated_data (numpy array): The linearly interpolated data buffer.
        """
        L = len(data_buffer)
        resampling_point = L
        even_times = np.linspace(times[0], times[-1], resampling_point)
        interp = np.interp(even_times, times, data_buffer)
        interpolated_data = np.hamming(resampling_point) * interp
        return interpolated_data

    def fft(self, data_buffer, fps):
        """
        Compute the Fast Fourier Transform (FFT) of the input data buffer.

        Args:
            data_buffer (numpy array): The input data buffer.
            fps (float): The frame rate of the video.

        Returns:
            fft_of_interest (numpy array): The FFT values of interest.
            freqs_of_interest (numpy array): The corresponding frequencies of interest.
        """
        L = len(data_buffer)
        freqs = float(fps) / L * np.arange(L / 2 + 1)
        freqs_in_minute = 60. * freqs
        raw_fft = np.fft.rfft(data_buffer * 30)
        fft = np.abs(raw_fft) ** 2
        interest_idx = np.where((freqs_in_minute > 50) & (freqs_in_minute < 180))[0]
        interest_idx_sub = interest_idx[:-1].copy()
        freqs_of_interest = freqs_in_minute[interest_idx_sub]
        fft_of_interest = fft[interest_idx_sub]
        return fft_of_interest, freqs_of_interest

    def butter_bandpass_filter(self, data_buffer, lowcut, highcut, fs, order=5):
        """
        Apply a Butterworth bandpass filter to the input data buffer.

        Args:
            data_buffer (numpy array): The input data buffer.
            lowcut (float): The lower cutoff frequency.
            highcut (float): The higher cutoff frequency.
            fs (float): The sampling rate (frame rate).
            order (int): The order of the Butterworth filter.

        Returns:
            filtered_data (numpy array): The filtered data buffer.
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        coefficients = signal.butter(order, [low, high], 'bandpass')
        return signal.filtfilt(*coefficients, data_buffer)
