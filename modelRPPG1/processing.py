import numpy as np
from scipy import signal

class Signal_processing():
    def __init__(self):
        self.a = 1

    def extract_color(self, ROIs):
        """
        Extract average value of green color from ROIs.

        Args:
            ROIs (tuple): Tuple of regions of interest (ROIs).

        Returns:
            float: Average green value.
        """
        g = []
        for ROI in ROIs:
            g.append(np.mean(ROI[:, :, 1]))
        output_val = np.mean(g)
        return output_val

    def calculate_r_ir_ratio(self, roi_data, brightest_pixels_ratio=0.1):
        """
        Calculate the Ratio-of-Ratios (R/IR) from the ROI data using the brightest pixels.

        Args:
            roi_data (tuple): Tuple of regions of interest (ROIs) containing red, green, and blue channel data.
            brightest_pixels_ratio (float, optional): Ratio of brightest pixels to use. Defaults to 0.1 (10%).

        Returns:
            float: Ratio-of-Ratios (R/IR) value.
        """
        red_values = []
        ir_values = []
        for roi in roi_data:
            red_roi = roi[:, :, 2]
            ir_roi = roi[:, :, 1]
            red_roi_flat = red_roi.flatten()
            ir_roi_flat = ir_roi.flatten()
            red_roi_flat.sort()
            ir_roi_flat.sort()
            num_pixels = len(red_roi_flat)
            red_values.append(np.mean(red_roi_flat[-int(num_pixels * brightest_pixels_ratio):]))
            ir_values.append(np.mean(ir_roi_flat[-int(num_pixels * brightest_pixels_ratio):]))
        r_ir_ratio = np.mean(red_values) / np.mean(ir_values)
        return r_ir_ratio

    def normalization(self, data_buffer):
        """
        Normalize the input data buffer.

        Args:
            data_buffer (numpy.ndarray): Input data buffer.

        Returns:
            numpy.ndarray: Normalized data buffer.
        """
        normalized_data = data_buffer / np.linalg.norm(data_buffer)
        return normalized_data

    def signal_detrending(self, data_buffer):
        """
        Remove overall trending from the data buffer.

        Args:
            data_buffer (numpy.ndarray): Input data buffer.

        Returns:
            numpy.ndarray: Detrended data buffer.
        """
        detrended_data = signal.detrend(data_buffer)
        return detrended_data

    def interpolation(self, data_buffer, times, window='hamming'):
        """
        Interpolate data buffer to make the signal more periodic (avoid spectral leakage).

        Args:
            data_buffer (numpy.ndarray): Input data buffer.
            times (numpy.ndarray): Corresponding timestamps for the data buffer.
            window (str, optional): Window function to apply. Defaults to 'hamming'.

        Returns:
            numpy.ndarray: Interpolated data buffer.
        """
        L = len(data_buffer)
        even_times = np.linspace(times[0], times[-1], L)
        interp = np.interp(even_times, times, data_buffer)
        if window == 'hamming':
            interpolated_data = np.hamming(L) * interp
        elif window == 'hanning':
            interpolated_data = np.hanning(L) * interp
        elif window == 'blackman':
            interpolated_data = np.blackman(L) * interp
        else:
            interpolated_data = interp
        return interpolated_data

    def fft(self, data_buffer, fps):
        """
        Perform Fast Fourier Transform (FFT) on the data buffer.

        Args:
            data_buffer (numpy.ndarray): Input data buffer.
            fps (float): Frames per second.

        Returns:
            numpy.ndarray, numpy.ndarray: FFT of interest and corresponding frequencies.
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
        Apply Butterworth bandpass filter to the data buffer.

        Args:
            data_buffer (numpy.ndarray): Input data buffer.
            lowcut (float): Lower cutoff frequency.
            highcut (float): Higher cutoff frequency.
            fs (float): Sampling frequency.
            order (int, optional): Filter order. Defaults to 5.

        Returns:
            numpy.ndarray: Filtered data buffer.
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        coefficients = signal.butter(order, [low, high], 'bandpass')
        return signal.filtfilt(*coefficients, data_buffer)

    def calibrate_spO2(self, estimated_spO2):
        """
        Calibrate the estimated SpO2 for the individual.

        Args:
            estimated_spO2 (float): Estimated SpO2 value.

        Returns:
            float: Calibrated SpO2 value.
        """
        # Adjust the estimated SpO2 based on individual characteristics
        calibrated_spO2 = estimated_spO2 * 0.95 + 2.5
        return calibrated_spO2
