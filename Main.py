import numpy as np
import cv2 as cv2
try:
    import dlib
    print("Dlib is installed.")
    print("Dlib version:", dlib.__version__)
except ImportError:
    print("Dlib is not installed.")

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
       print(face)

    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

