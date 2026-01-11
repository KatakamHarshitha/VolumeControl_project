import cv2
import mediapipe as mp
import pyautogui
from math import hypot
import numpy as np
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

last_action_time = 0
action_delay = 0.25

camera_on = False
volume_status = "Stable"
status_color = (255, 255, 255)

START_BTN = (20, 520, 140, 570)
STOP_BTN  = (160, 520, 280, 570)

def draw_ui(img):
    
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (300, img.shape[0]), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)

    
    cv2.putText(img, "Gesture Volume Control",
                (20, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8, (0, 255, 0), 2)


    cam_text = "ON" if camera_on else "OFF"
    cam_color = (0, 255, 0) if camera_on else (0, 0, 255)
    cv2.putText(img, f"Camera : {cam_text}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, cam_color, 2)

    
    cv2.rectangle(img, START_BTN[:2], START_BTN[2:], (0, 200, 0), -1)
    cv2.rectangle(img, STOP_BTN[:2], STOP_BTN[2:], (0, 0, 200), -1)

    cv2.putText(img, "START",
                (40, 555),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    cv2.putText(img, "STOP",
                (185, 555),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

def mouse_click(event, x, y, flags, param):
    global camera_on
    if event == cv2.EVENT_LBUTTONDOWN:
        if START_BTN[0] < x < START_BTN[2] and START_BTN[1] < y < START_BTN[3]:
            camera_on = True
        if STOP_BTN[0] < x < STOP_BTN[2] and STOP_BTN[1] < y < STOP_BTN[3]:
            camera_on = False

cv2.namedWindow("Enhanced Gesture Volume Control")
cv2.setMouseCallback("Enhanced Gesture Volume Control", mouse_click)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1000, 600))
    draw_ui(frame)

    if camera_on:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                lm_list = []
                h, w, c = frame.shape

                for id, lm in enumerate(hand_landmark.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])

                x1, y1 = lm_list[4][1], lm_list[4][2]
                x2, y2 = lm_list[8][1], lm_list[8][2]

                cv2.circle(frame, (x1, y1), 8, (255, 0, 255), -1)
                cv2.circle(frame, (x2, y2), 8, (255, 0, 255), -1)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                length = hypot(x2 - x1, y2 - y1)
                vol_per = int(np.clip(np.interp(length, [30, 200], [0, 100]), 0, 100))
                vol_bar = np.interp(length, [30, 200], [450, 200])

                # Status
                if length > 140:
                    volume_status = "Increasing"
                    status_color = (0, 255, 0)
                elif length < 40:
                    volume_status = "Decreasing"
                    status_color = (0, 0, 255)
                else:
                    volume_status = "Stable"
                    status_color = (255, 255, 0)

                current_time = time.time()
                if current_time - last_action_time > action_delay:
                    if length > 140:
                        pyautogui.press("volumeup")
                        last_action_time = current_time
                    elif length < 40:
                        pyautogui.press("volumedown")
                        last_action_time = current_time

                # Volume bar
                cv2.rectangle(frame, (50, 200), (80, 450), (80, 80, 80), 2)
                cv2.rectangle(frame, (50, int(vol_bar)), (80, 450), (0, 255, 0), -1)

                # Text
                cv2.putText(frame, f"Volume : {vol_per}%",
                            (100, 230),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

                cv2.putText(frame, f"Gesture : {volume_status}",
                            (100, 270),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, status_color, 2)

                mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

    else:
        cv2.putText(frame, "Camera is OFF",
                    (400, 300),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1, (0, 0, 255), 3)

    cv2.imshow("Enhanced Gesture Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
