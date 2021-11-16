import os,sys 
sys.path.append(os.path.dirname(__file__))

import cv2 
import mediapipe as mp

def draw_pose_landmarks(
    image,
    landmarks,
    visibility_th=0.5,
    thickness = 2
):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_th:
            continue

        if index == 0:  # 鼻
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 1:  # 右目：目頭
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 2:  # 右目：瞳
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 3:  # 右目：目尻
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 4:  # 左目：目頭
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 5:  # 左目：瞳
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 6:  # 左目：目尻
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 7:  # 右耳
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 8:  # 左耳
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 9:  # 口：左端
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 10:  # 口：左端
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 11:  # 右肩
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 12:  # 左肩
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 13:  # 右肘
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 14:  # 左肘
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 15:  # 右手首
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 16:  # 左手首
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 17:  # 右手1(外側端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 18:  # 左手1(外側端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 19:  # 右手2(先端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 20:  # 左手2(先端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 21:  # 右手3(内側端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 22:  # 左手3(内側端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 23:  # 腰(右側)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 24:  # 腰(左側)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 25:  # 右ひざ
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 26:  # 左ひざ
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 27:  # 右足首
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 28:  # 左足首
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 29:  # 右かかと
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 30:  # 左かかと
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 31:  # 右つま先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)
        if index == 32:  # 左つま先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), thickness)

        # if not upper_body_only:
        if False:
            cv2.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                       cv2.LINE_AA)

    if len(landmark_point) > 0:
        # 右目
        if landmark_point[1][0] > visibility_th and landmark_point[2][0] > visibility_th:
            cv2.line(image, landmark_point[1][1], landmark_point[2][1],(0, 255, 0), thickness)
        if landmark_point[2][0] > visibility_th and landmark_point[3][0] > visibility_th:
            cv2.line(image, landmark_point[2][1], landmark_point[3][1],(0, 255, 0), thickness)

        # 左目
        if landmark_point[4][0] > visibility_th and landmark_point[5][0] > visibility_th:
            cv2.line(image, landmark_point[4][1], landmark_point[5][1],(0, 255, 0), thickness)
        if landmark_point[5][0] > visibility_th and landmark_point[6][0] > visibility_th:
            cv2.line(image, landmark_point[5][1], landmark_point[6][1],(0, 255, 0), thickness)

        # 口
        if landmark_point[9][0] > visibility_th and landmark_point[10][0] > visibility_th:
            cv2.line(image, landmark_point[9][1], landmark_point[10][1],(0, 255, 0), thickness)

        # 肩
        if landmark_point[11][0] > visibility_th and landmark_point[12][0] > visibility_th:
            cv2.line(image, landmark_point[11][1], landmark_point[12][1],(0, 255, 0), thickness)

        # 右腕
        if landmark_point[11][0] > visibility_th and landmark_point[13][0] > visibility_th:
            cv2.line(image, landmark_point[11][1], landmark_point[13][1],(0, 255, 0), thickness)
        if landmark_point[13][0] > visibility_th and landmark_point[15][0] > visibility_th:
            cv2.line(image, landmark_point[13][1], landmark_point[15][1],(0, 255, 0), thickness)

        # 左腕
        if landmark_point[12][0] > visibility_th and landmark_point[14][0] > visibility_th:
            cv2.line(image, landmark_point[12][1], landmark_point[14][1],(0, 255, 0), thickness)
        if landmark_point[14][0] > visibility_th and landmark_point[16][0] > visibility_th:
            cv2.line(image, landmark_point[14][1], landmark_point[16][1],(0, 255, 0), thickness)

        # 右手
        if landmark_point[15][0] > visibility_th and landmark_point[17][0] > visibility_th:
            cv2.line(image, landmark_point[15][1], landmark_point[17][1],(0, 255, 0), thickness)
        if landmark_point[17][0] > visibility_th and landmark_point[19][0] > visibility_th:
            cv2.line(image, landmark_point[17][1], landmark_point[19][1],(0, 255, 0), thickness)
        if landmark_point[19][0] > visibility_th and landmark_point[21][0] > visibility_th:
            cv2.line(image, landmark_point[19][1], landmark_point[21][1],(0, 255, 0), thickness)
        if landmark_point[21][0] > visibility_th and landmark_point[15][0] > visibility_th:
            cv2.line(image, landmark_point[21][1], landmark_point[15][1],(0, 255, 0), thickness)

        # 左手
        if landmark_point[16][0] > visibility_th and landmark_point[18][0] > visibility_th:
            cv2.line(image, landmark_point[16][1], landmark_point[18][1],(0, 255, 0), thickness)
        if landmark_point[18][0] > visibility_th and landmark_point[20][0] > visibility_th:
            cv2.line(image, landmark_point[18][1], landmark_point[20][1],(0, 255, 0), thickness)
        if landmark_point[20][0] > visibility_th and landmark_point[22][0] > visibility_th:
            cv2.line(image, landmark_point[20][1], landmark_point[22][1],(0, 255, 0), thickness)
        if landmark_point[22][0] > visibility_th and landmark_point[16][0] > visibility_th:
            cv2.line(image, landmark_point[22][1], landmark_point[16][1],(0, 255, 0), thickness)

        # 胴体
        if landmark_point[11][0] > visibility_th and landmark_point[23][0] > visibility_th:
            cv2.line(image, landmark_point[11][1], landmark_point[23][1],(0, 255, 0), thickness)
        if landmark_point[12][0] > visibility_th and landmark_point[24][0] > visibility_th:
            cv2.line(image, landmark_point[12][1], landmark_point[24][1],(0, 255, 0), thickness)
        if landmark_point[23][0] > visibility_th and landmark_point[24][0] > visibility_th:
            cv2.line(image, landmark_point[23][1], landmark_point[24][1],(0, 255, 0), thickness)

        if len(landmark_point) > 25:
            # 右足
            if landmark_point[23][0] > visibility_th and landmark_point[25][0] > visibility_th:
                cv2.line(image, landmark_point[23][1], landmark_point[25][1],(0, 255, 0), thickness)
            if landmark_point[25][0] > visibility_th and landmark_point[27][0] > visibility_th:
                cv2.line(image, landmark_point[25][1], landmark_point[27][1],(0, 255, 0), thickness)
            if landmark_point[27][0] > visibility_th and landmark_point[29][0] > visibility_th:
                cv2.line(image, landmark_point[27][1], landmark_point[29][1],(0, 255, 0), thickness)
            if landmark_point[29][0] > visibility_th and landmark_point[31][0] > visibility_th:
                cv2.line(image, landmark_point[29][1], landmark_point[31][1],(0, 255, 0), thickness)

            # 左足
            if landmark_point[24][0] > visibility_th and landmark_point[26][0] > visibility_th:
                cv2.line(image, landmark_point[24][1], landmark_point[26][1],(0, 255, 0), thickness)
            if landmark_point[26][0] > visibility_th and landmark_point[28][0] > visibility_th:
                cv2.line(image, landmark_point[26][1], landmark_point[28][1],(0, 255, 0), thickness)
            if landmark_point[28][0] > visibility_th and landmark_point[30][0] > visibility_th:
                cv2.line(image, landmark_point[28][1], landmark_point[30][1],(0, 255, 0), thickness)
            if landmark_point[30][0] > visibility_th and landmark_point[32][0] > visibility_th:
                cv2.line(image, landmark_point[30][1], landmark_point[32][1],(0, 255, 0), thickness)
    return image

def draw_hand_landmarks(
    image,
    landmarks
):
    image_height,image_width = image.shape[0],image.shape[1]
    landmark_point = list()

    for landmark in landmarks.landmark:
	    landmark_x = int(landmark.x*image_width)
	    landmark_y = int(landmark.y*image_height)

	    landmark_point.append([landmark_x,landmark_y])

    if len(landmark_point) > 0:
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),(0, 0, 0),6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),(255, 255, 255),2)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),(255, 255, 255), 2)

        # 人差指
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),(255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),(255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),(255, 255, 255), 2)

        # 中指
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),(255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),(255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),(255, 255, 255), 2)

        # 薬指
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),(255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),(255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),(255, 255, 255), 2)

        # 小指
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),(255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),(255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),(255, 255, 255), 2)

        # 手の平
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),(255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),(255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),(255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),(255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),(255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),(255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),(0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),(255, 255, 255), 2)

    # キーポイント
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),-1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),-1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),-1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image
        