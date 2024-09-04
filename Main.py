import cv2
import mediapipe as mp
import pyautogui as pyi
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
    
    if lankmarks:
        lank = lankmarks[0].landmark
        for id,l in enumerate(lank[474:478]):
            x = int( l.x * frame_w)
            y = int (l.y * frame_h)
            cv2.circle(frame,(x,y),3,(0,255,0))
            if id == 1:
                screen_x = screen_w / frame_w * x
                screen_y = screen_h / frame_h * y
              #  pyi.moveTo(screen_x,screen_y)
        left = [lank[145],lank[159]]
        for l in left:
            x = int( l.x * frame_w)
            y = int (l.y * frame_h)
            cv2.circle(frame,(x,y),3,(0,255,255))
        
        if(left[0].y-left[1].y) < 0.004:
            print('Eye Is blinked')
          
            pyi.click()
            pyi.sleep(1)


        if(left[0].y-left[1].y) < 0.000:
            print('Eye Is closed')
            print('..........................')
            pyi.sleep(1)
            




    cv2.imshow("Eye Controlled Mouse",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cam.release()
cv2.destroyAllWindows()
