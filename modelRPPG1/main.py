import cv2
import numpy as np
import time
import pyqtgraph as pg
import mediapipe as mp
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
from processing import Signal_processing

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize processing flag
video = False

# Initialize signal processing object
sp = Signal_processing()

# Define cheek landmarks
cheeks = [329, 346, 426, 100, 117, 206]

# Initialize counters
i = 0
t = time.time()

# Set buffer size for signal processing
BUFFER_SIZE = 300

# Initialize FPS counter
fps = 0

# Initialize data buffers
times = []
data_buffer = []

# Initialize plotting variables
filtered_data = []
fft_of_interest = []
freqs_of_interest = []
bpm = 0

# Set up plotting
app = QApplication([])
win = pg.GraphicsLayoutWidget(title='Signals')
p = win.addPlot(title="Green Channel Signal")
p1 = win.addPlot(title="Detrended Signal")
p2 = win.addPlot(title="Filtered Signal")
win.setBackground('white')
win.resize(1200, 400)
win.show()

def update():
    # Update plots
    p.clear()
    p.plot(data_buffer, pen={'color': 'g', 'width': 2})

    p1.clear()
    p1.plot(np.column_stack((freqs_of_interest, fft_of_interest)), pen={'color': 'g', 'width': 2})

    p2.clear()
    p2.plot(filtered_data[20:], pen={'color': 'g', 'width': 2})

    app.processEvents()

# Set up timer for updating plots
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(300)  # Update graph every 300ms

# Initialize Mediapipe face mesh
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec_w = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
drawSpec_b = mpDraw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)

while True:
    # Capture frame
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process face mesh
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # Draw face mesh
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec_b, drawSpec_w)

            # Extract cheek landmarks
            landmark = []
            for id, lm in enumerate(faceLms.landmark):
                if id in cheeks:
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    landmark.append([id, x, y])

            # Define ROIs for cheeks
            ROIs = (img[landmark[3][2]:landmark[5][2], landmark[3][1]:landmark[4][1]],
                    img[landmark[0][2]:landmark[2][2], landmark[1][1]:landmark[0][1]])
            cv2.rectangle(img, (landmark[3][1], landmark[3][2]), (landmark[4][1], landmark[5][2]), (0, 255, 0), 1)
            cv2.rectangle(img, (landmark[1][1], landmark[0][2]), (landmark[0][1], landmark[2][2]), (0, 255, 0), 1)

    # Extract average green value from ROIs
    green_val = sp.extract_color(ROIs)
    data_buffer.append(green_val)

    # Calculate R/IR ratio using the brightest pixels
    r_ir_ratio = sp.calculate_r_ir_ratio(ROIs, 0.2)  # Use the brightest 20% of pixels

    # Estimate SpO2 from R/IR ratio using a more accurate equation
    spO2 = 100 - 25 * (r_ir_ratio - 0.4)
    spO2 = max(0, min(100, spO2))  # Ensure SpO2 is within the valid range

    # Calibrate SpO2 estimation for the individual
    spO2 = sp.calibrate_spO2(spO2)

    # Append timestamp for real-time processing
    if video == False:
        times.append(time.time() - t)

    # Manage data buffer size
    L = len(data_buffer)
    if L > BUFFER_SIZE:
        data_buffer = data_buffer[-BUFFER_SIZE:]
        times = times[-BUFFER_SIZE:]
        L = BUFFER_SIZE

    # Process data when buffer is full
    if L == 300:
        # Calculate FPS
        fps = float(L) / (times[-1] - times[0])

        # Process signal
        detrended_data = sp.signal_detrending(data_buffer)
        filtered_data = sp.butter_bandpass_filter(detrended_data, 0.7, 4, fps, order=5)  # Adjust filter order
        interpolated_data = sp.interpolation(filtered_data, times, window='hanning')  # Try different windows
        normalized_data = sp.normalization(interpolated_data)
        fft_of_interest, freqs_of_interest = sp.fft(normalized_data, fps)

        # Calculate heart rate
        max_arg = np.argmax(fft_of_interest)
        bpm = freqs_of_interest[max_arg]

        # Display FPS, SpO2 e HR values
        cv2.putText(img, "FPS: {0:.2f}".format(fps), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, "SpO2: {0:.2f}%".format(spO2), (30, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, "HR: {} bpm".format(round(bpm)), (30, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # Display frame
    cv2.imshow("Image", img)

    # Handle user input
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    i += 1

cap.release()
cv2.destroyAllWindows()
