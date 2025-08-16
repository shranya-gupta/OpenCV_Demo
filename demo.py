import cv2
import numpy as np
import datetime as dt

frame_width = 640
frame_height = 480
cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)
cap.set(10, 50)  # brightness

def empty(x):
    pass

cv2.namedWindow('parameters')
cv2.resizeWindow('parameters', 640, 240)
cv2.createTrackbar('threshold1', 'parameters', 100, 255, empty)
cv2.createTrackbar('threshold2', 'parameters', 200, 255, empty)
cv2.createTrackbar('Area', 'parameters', 1000, 20000, empty)

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def getContours(imgDilate, imgContours):
    contours, _ = cv2.findContours(cv2.cvtColor(imgDilate, cv2.COLOR_BGR2GRAY),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area_min = cv2.getTrackbarPos('Area', 'parameters')
        area = cv2.contourArea(cnt)
        if area > area_min:
            cv2.drawContours(imgContours, cnt, -1, (0, 255, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContours, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cv2.putText(imgContours, "Points: " + str(len(approx)),
                        (x + w + 20, y + h + 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(imgContours, 'Area: ' + str(int(area)),
                        (x + w + 45, y + h + 45),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

# MODE CONTROL
mode = "normal"  # can be: normal / contours / face

print("Press 'c' = Contours mode | 'f' = Face detection mode | 'n' = Normal mode | 'q' = Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    imgContours = frame.copy()
    key = cv2.waitKey(1) & 0xFF

    # Change mode
    if key == ord('c'):
        mode = "contours"
    elif key == ord('f'):
        mode = "face"
    elif key == ord('n'):
        mode = "normal"
    elif key == ord('q'):
        break

    # MODE: Contour detection
    if mode == "contours":
        imgBlur = cv2.GaussianBlur(frame, (7, 7), 1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        t1 = cv2.getTrackbarPos('threshold1', 'parameters')
        t2 = cv2.getTrackbarPos('threshold2', 'parameters')
        imgCanny = cv2.Canny(imgGray, t1, t2)
        imgCanny = cv2.cvtColor(imgCanny, cv2.COLOR_GRAY2BGR)
        kernel = np.ones((5, 5), np.uint8)
        imgDilate = cv2.dilate(imgCanny, kernel, iterations=1)
        getContours(imgDilate, imgContours)
        output = np.hstack([imgDilate, imgContours])
        cv2.imshow("Output", output)

    # MODE: Face detection
    elif mode == "face":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face Detection Mode", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Output", frame)

    # MODE: Normal with Name & Date/Time
    else:
        # Top-right for your name
        cv2.putText(frame, "PERSON_NAME", (frame.shape[1] - 250, 50),
                    cv2.FONT_ITALIC, 1, (180, 0, 0), 2)

        # Top-left for mode name
        cv2.putText(frame, "Normal Mode", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Bottom-left for date and time
        date_time = str(dt.datetime.now())
        cv2.putText(frame, date_time, (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 0), 2)

        cv2.imshow("Output", frame)

cap.release()
cv2.destroyAllWindows()
