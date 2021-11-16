import os,sys 
sys.path.append(os.path.dirname(__file__))

from .tools import draw_pose_landmarks,draw_hand_landmarks 
import mediapipe as mp 
import cv2


def cam_pose():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence = 0.7,
        min_tracking_confidence = 0.5
    )

    capture = cv2.VideoCapture(0)


    while True:
        ret,frame = capture.read()
        debug_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = pose.process(debug_image)
        if results.pose_landmarks is not None:
            frame = draw_pose_landmarks(
                frame,
                results.pose_landmarks,

            )
        cv2.imshow('result',frame)

        k = cv2.waitKey(1)
        if k == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

def cam_hands():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence = 0.7,
        min_tracking_confidence = 0.5
    )

    capture = cv2.VideoCapture(0)

    while True:
        ret,frame = capture.read()
        window_size = (800,600)
        #frame = cv2.resize(frame,window_size)

        debug_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #print(type(image))
        results = hands.process(debug_image)
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                frame = draw_hand_landmarks(frame,hand_landmarks)
        
        cv2.imshow('result',frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    capture.release()
    cv2.destroyAllWindows() 