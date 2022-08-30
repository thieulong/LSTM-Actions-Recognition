from cProfile import label
import cv2
import mediapipe as mp
import pandas as pd
import numpy
import keras
import threading

cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = keras.models.load_model("lstm-model.h5")

lm_list = []

def make_landmark_timestep(results):
    print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, frame):
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = frame.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,30)
    fontScale = 1
    if label == "punch":
        fontColor = (0,0,255)
    else:
        fontColor = (0,255,0)
    thickness = 2
    lineType = 2
    cv2.putText(img, str(label),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(model, lm_list):
    global label
    lm_list = numpy.array(lm_list)
    lm_list = numpy.expand_dims(lm_list, axis=0)
    result = model.predict(lm_list)
    if result[0][0] > 0.5:
        label = "punch"
    else:
        label = "neutral"
    return str(label)

i = 0
warm_up_frames = 60

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    i=i+1
    if i > warm_up_frames:
        print("Start detecting...")
        if results.pose_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            if len(lm_list) == 20:
                t1 = threading.Thread(target=detect, args=(model, lm_list, ))
                t1.start()
                lm_list = []
            x_coordinate = list()
            y_coordinate = list()
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_coordinate.append(cx)
                y_coordinate.append(cy)
            if label == "neutral":
                cv2.rectangle(img=frame,
                                pt1=(min(x_coordinate), max(y_coordinate)),
                                pt2=(max(x_coordinate), min(y_coordinate)-25),
                                color=(0,255,0),
                                thickness=1)
            elif label == "punch":
                cv2.rectangle(img=frame,
                                pt1=(min(x_coordinate), max(y_coordinate)),
                                pt2=(max(x_coordinate), min(y_coordinate)-25),
                                color=(0,0,255),
                                thickness=3)

            frame = draw_landmark_on_image(mpDraw, results, frame)
        frame = draw_class_on_image(label, frame)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

df = pd.DataFrame(lm_list)
df.to_csv(label+".txt")
cap.release()
cv2.destroyAllWindows()