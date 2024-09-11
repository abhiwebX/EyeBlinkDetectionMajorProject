import numpy as np
import cv2
import mediapipe as mp
import pyautogui as pyi
import time

# Initialize camera and keyboard layout
cam = cv2.VideoCapture(0)
keyboard = np.zeros((800, 1370, 3), np.uint8)

# Dataset for letters
dataset = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "P", 10: "K", 11: "L", 12: "M"}

# Mediapipe FaceMesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyi.size()

# Keyboard display function
def letter(letter_index, tt, LetterL):
    # Dynamically calculate x and y positions based on index
    row = letter_index // 5  # 5 letters per row
    col = letter_index % 5   # Modulus to get column
    
    x = col * 200  # 200 pixels between each letter horizontally
    y = row * 200  # 200 pixels between each letter vertically
    
    width = 100
    height = 100
    border = 2
    
    # Draw rectangle (filled if it's the active letter)
    if LetterL:
        cv2.rectangle(keyboard, (x + border, y + border), (x + width - border, y + height - border), (255, 255, 255), -1)
    else:
        cv2.rectangle(keyboard, (x + border, y + border), (x + width - border, y + height - border), (255, 0, 0), border)
    
    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    text = tt
    font_scale = 7
    font_th = 2
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y
    cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (0, 255, 0), font_th)

# Eye blink detection variables
Kframes = 0
Kindex = 0
label_text = f"Event Name: {dataset[Kindex]}"  # Initialize the label with the first box

# Main loop
while True:
    _, frame = cam.read()
    keyboard[:] = (0, 0, 0)  # Reset keyboard display
    frame = cv2.flip(frame, 1)  # Flip for a mirror effect
    rgb_face = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    output = face_mesh.process(rgb_face)
    landmarks = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    
    leftEyecount = 0
    righteyecount = 0
    
    if landmarks:
        lank = landmarks[0].landmark
        
        # Right eye detection
        right = [lank[374], lank[385]]
        for l in right:
            x = int(l.x * frame_w)
            y = int(l.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        if (right[0].y - right[1].y) < 0.005:
            righteyecount = 1  # Right eye is blinked
        
        # Left eye detection
        left = [lank[145], lank[159]]
        for l in left:
            x = int(l.x * frame_w)
            y = int(l.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        if (left[0].y - left[1].y) < 0.005:
            leftEyecount = 1  # Left eye is blinked

    # Eye blink patterns
    if righteyecount == 1 and leftEyecount == 1:
        print("Double blink pattern detected")
        # Add functionality for double blink here
    elif righteyecount == 1 and leftEyecount == 0:
        print("Right blink detected (box changing)")
        # Move selection to next box
        Kframes += 1  
    elif leftEyecount == 1 and righteyecount == 0:
        print("Event Blink")
        # Update the label text with the current box name on event blink
        label_text = f"Event Name: {dataset[Kindex]}"  # Update label text

    # Display selected character in the keyboard
    if Kframes == 8:
        Kindex += 1
        Kframes = 0
    if Kindex == 13:
        Kindex = 0

    # Draw keyboard and highlight selected box
    for i in range(13):
        if i == Kindex:
            light = True
        else:
            light = False
        letter(i, dataset[i], light)

    # Resize and insert the camera frame into the top-right corner of the keyboard
    resized_frame = cv2.resize(frame, (300, 200))  # Resize webcam frame
    keyboard[0:200, 1070:1370] = resized_frame  # Place webcam frame in top-right corner with margin
    
    # Display the label below the camera frame
    cv2.putText(keyboard, label_text, (1080, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display the keyboard and the frame
    cv2.imshow("Virtual Keyboard", keyboard)

    # Exit on pressing the Escape key or 'q' key
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

# Release resources
time.sleep(2)
cam.release()
cv2.destroyAllWindows()
