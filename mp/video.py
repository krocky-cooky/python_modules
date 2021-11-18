import os,sys 
sys.path.append(os.path.dirname(__file__))

from .tools import draw_pose_landmarks,draw_hand_landmarks 
import mediapipe as mp 
import cv2
from tqdm import tqdm


def hand_video(
    path = 'sample.mov',
    output_path = 'output.mp4',
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.5
):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence = min_detection_confidence,
        min_tracking_confidence = min_tracking_confidence
    )
    capture = cv2.VideoCapture(path)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    #フレームレート(1フレームの時間単位はミリ秒)の取得
    frame_rate = int(capture.get(cv2.CAP_PROP_FPS))

    fmt = cv2.VideoWriter_fourcc('m','p','4','v')
    writer = cv2.VideoWriter(output_path,fmt,frame_rate,size)

    if not capture.isOpened():
        raise Exception('Could not read video')

    for _ in tqdm(range(frame_count)):
        ret,frame = capture.read()

        if not ret:
            break 

        debug_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = hands.process(debug_image)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                frame = draw_hand_landmarks(frame,hand_landmarks)
        
        writer.write(frame)

    capture.release()
    print('done')



def pose_video(
    path = 'sample.mov',
    output_path = 'output.mp4',
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.5
):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence = min_detection_confidence,
        min_tracking_confidence = min_tracking_confidence
    )
    capture = cv2.VideoCapture(path)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    #フレームレート(1フレームの時間単位はミリ秒)の取得
    frame_rate = int(capture.get(cv2.CAP_PROP_FPS))

    fmt = cv2.VideoWriter_fourcc('m','p','4','v')
    writer = cv2.VideoWriter(output_path,fmt,frame_rate,size)

    if not capture.isOpened():
        raise Exception('Could not read video')

    for _ in tqdm(range(frame_count)):
        ret,frame = capture.read()


        debug_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = pose.process(debug_image)

        if results.pose_landmarks is not None:
            frame = draw_pose_landmarks(
                frame,
                results.pose_landmarks,

            )
        
        writer.write(frame)

    capture.release()
    print('done')