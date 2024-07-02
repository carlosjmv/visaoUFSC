import cv2
import numpy as np
import time
import pyqtgraph as pg
import mediapipe as mp
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
from scipy.signal import savgol_filter
from processing import Signal_processing
from skin_detector import skin_detector # From https://github.com/WillBrennan/SkinDetector

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils


def main(user_input=None):
    BUFFER_SIZE = 300  # Size of the buffer for storing mean RGB values
    sp = Signal_processing()  # Instance of the Signal_processing class
    framerate = 30  # Assumed frame rate of the camera
    t = time.time()  # Start time for calculating FPS
    times = []  # List to store the time elapsed between frames
    fft_of_interest = []  # List to store the FFT values of interest
    freqs_of_interest = []  # List to store the frequencies of interest

    # Plotting setup
    app = QApplication([])
    win = pg.GraphicsLayoutWidget(title="Signals")
    p1 = win.addPlot(title="Detrended Signal")  # Plot for the detrended signal
    p2 = win.addPlot(title="Filtered & Savitzky-Golay Smoothed Signal")  # Plot for the filtered and smoothed signal
    win.setBackground('white')
    win.resize(1400, 400)
    win.show()

    def update():
        """
        Update the plots with the latest data.
        """
        p1.clear()
        p1.plot(np.column_stack((freqs_of_interest, fft_of_interest)), pen={'color': 'g', 'width': 2})

        p2.clear()
        p2.plot(normalized_data[20:], pen={'color': 'g', 'width': 2})

        app.processEvents()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(300)  # Update the graphs every 300ms

    # Frequency analysis
    plot = False
    image_show = True
    camera = cv2.VideoCapture(0)

    # Loop on video frames
    frame_counter = 0

    with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:
        while True:
            (grabbed, frame) = camera.read()

            if not grabbed:
                continue

            h, w, _ = frame.shape
            # Convert frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)
            # Convert back to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # If no face detected, read next frame
            if not (results.detections):
                continue

            if image_show:
                show_frame = frame.copy()

            if results.detections:
                for id, detection in enumerate(results.detections):
                    drawing_spec = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    mp_draw.draw_detection(image, detection, drawing_spec, drawing_spec)
                    bBox = detection.location_data.relative_bounding_box
                    h, w, c = image.shape

                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                    ROI = image[boundBox[1]:boundBox[1] + boundBox[3], boundBox[0]:boundBox[0] + boundBox[2]]

                mask = skin_detector.process(ROI)

                masked_face = cv2.bitwise_and(ROI, ROI, mask=mask)
                # Count only > 0 value (not black)
                number_of_skin_pixels = np.sum(mask > 0)

                # Compute mean RGB values
                r = np.sum(masked_face[:, :, 2]) / number_of_skin_pixels
                g = np.sum(masked_face[:, :, 1]) / number_of_skin_pixels
                b = np.sum(masked_face[:, :, 0]) / number_of_skin_pixels

                if frame_counter == 0:
                    mean_rgb = np.array([r, g, b])
                else:
                    mean_rgb = np.vstack((mean_rgb, np.array([r, g, b])))

            times.append(time.time() - t)

            L = len(mean_rgb)

            if L > BUFFER_SIZE:
                mean_rgb = mean_rgb[-BUFFER_SIZE:]
                times = times[-BUFFER_SIZE:]
                L = BUFFER_SIZE

            if L == 300:  # If we have 300 frames
                fps = float(L) / (times[-1] - times[0])

                # Perform signal processing
                projection_matrix = np.array([[3, -2, 0], [1.5, 1, -1.5]])
                X_Y = np.matmul(projection_matrix, mean_rgb.T)
                std = np.array([1, -np.std(X_Y[0, :]) / np.std(X_Y[1, :])])
                S = np.matmul(std, X_Y)
                S = S / np.std(S)
                S = S.tolist()
                detrended_data = sp.signal_detrending(S)

                filtered_data = sp.butter_bandpass_filter(detrended_data, 0.7, 4, fps, order=3)

                # Perform linear interpolation
                interpolated_data = sp.interpolation(filtered_data, times)
                savitskized_data = savgol_filter(interpolated_data, 11, 3)
                normalized_data = sp.normalization(savitskized_data)

                # Compute the FFT of the processed signal
                fft_of_interest, freqs_of_interest = sp.fft(normalized_data, fps)

                # Estimate the heart rate
                max_arg = np.argmax(fft_of_interest)
                bpm = freqs_of_interest[max_arg]

                # Display FPS, SpO2 e HR values
                cv2.putText(image, "FPS: {0:.2f}".format(fps), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, f'SpO2: {int(detection.score[0] * 100)}%', (30, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "HR: {} bpm".format(round(bpm)), (30, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            # Display the processed frames
            cv2.imshow('Image', image)
            cv2.imshow('ROI', ROI)
            cv2.imshow("Masked face", masked_face)

            # Handle user input
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            frame_counter += 1

        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
