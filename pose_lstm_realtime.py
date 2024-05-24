import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tensorflow as tf
import threading
import h5py
import json

cap = cv2.VideoCapture(6)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

custom_objects = {
    'Orthogonal': tf.keras.initializers.Orthogonal
}

with h5py.File("lstm-model.h5", 'r') as f:
    model_config = f.attrs.get('model_config')
    model_config = json.loads(model_config)  

    for layer in model_config['config']['layers']:
        if 'time_major' in layer['config']:
            del layer['config']['time_major']

    model_json = json.dumps(model_config)

    model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)

    weights_group = f['model_weights']
    for layer in model.layers:
        layer_name = layer.name
        if layer_name in weights_group:
            weight_names = weights_group[layer_name].attrs['weight_names']
            layer_weights = [weights_group[layer_name][weight_name] for weight_name in weight_names]
            layer.set_weights(layer_weights)

lm_list = []
label = "neutral"
neutral_label = "neutral"

def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, frame):
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for lm in results.pose_landmarks.landmark:
        h, w, _ = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0) if label == neutral_label else (0, 0, 255)
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
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    result = model.predict(lm_list)
    if result[0][0] > 0.5:
        label = "violent" 
    else:
        label = "neutral"
    return str(label)

i = 0
warm_up_frames = 60

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    i += 1
    if i > warm_up_frames:
        if results.pose_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            if len(lm_list) == 20:
                t1 = threading.Thread(target=detect, args=(model, lm_list))
                t1.start()
                lm_list = []
            x_coordinate = []
            y_coordinate = []
            for lm in results.pose_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_coordinate.append(cx)
                y_coordinate.append(cy)
            cv2.rectangle(frame,
                            (min(x_coordinate), max(y_coordinate)),
                            (max(x_coordinate), min(y_coordinate) - 25),
                            (0, 255, 0),
                            1)

            frame = draw_landmark_on_image(mpDraw, results, frame)
        frame = draw_class_on_image(label, frame)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")
cap.release()
cv2.destroyAllWindows()
