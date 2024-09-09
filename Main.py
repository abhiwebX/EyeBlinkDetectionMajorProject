import numpy as np
import cv2
import mediapipe as mp
import pyautogui as pyi
import time


cam = cv2.VideoCapture(0)

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w , screen_h = pyi.size()


while True:
    _, frame = cam.read()
    frame = cv2.flip(frame,1)
    rgb_face = cv2.cvtColor(frame,cv2.COLOR_HSV2BGR)
    output = face_mesh.process(rgb_face)
    lankmarks = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    leftEyecount = 0
    righteyecount = 0
    if lankmarks:
        lank = lankmarks[0].landmark
        for id,l in enumerate(lank[474:478]):
            x = int( l.x * frame_w)
            y = int (l.y * frame_h)
        ##    cv2.circle(frame,(x,y),3,(0,255,0))

        right = [lank[374],lank[385]]
        for l in right:
             x = int( l.x * frame_w)
             y = int (l.y * frame_h)
             cv2.circle(frame,(x,y),3,(0,255,255))

        if(right[0].y-right[1].y) < 0.005:
            righteyecount = 1           
            pyi.sleep(4)
      
        left = [lank[145],lank[159]]
        for l in left:
            x = int( l.x * frame_w)
            y = int (l.y * frame_h)
            cv2.circle(frame,(x,y),3,(0,255,255))
        
        if(left[0].y-left[1].y) < 0.005:
            leftEyecount = 1
           
            pyi.sleep(4)
  
      

    cv2.imshow("Eye Controlled Mouse",frame)
        
    if(righteyecount == 1 and leftEyecount == 1):
        print("eye is blinked both eyes are blinked")
    elif(righteyecount == 1 and leftEyecount == 0):
        print(" Right eye is blinked event is performed")
    elif(leftEyecount == 1 and righteyecount==0):
        print("eye is blinked both eyes are blinked")

    key = cv2.waitKey(1)
    if key == 27:
        break


time.sleep(2)    
cam.release()
cv2.destroyAllWindows()
