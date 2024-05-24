import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tensorflow as tf
import threading
import h5py
import json

cap = cv2.VideoCapture(8)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, -10)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

custom_objects = {
    'Orthogonal': tf.keras.initializers.Orthogonal
}

with h5py.File("lstm-hand-gripping.h5", 'r') as f:
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

def make_landmark_timestep(results):
    c_lm = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                c_lm.append(lm.x)
                c_lm.append(lm.y)
                c_lm.append(lm.z)
    return c_lm

def draw_landmark_on_image(mpDraw, results, frame):
    for hand_landmarks in results.multi_hand_landmarks:
        mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
    return frame

def draw_bounding_box_and_label(frame, results, label):
    for hand_landmarks in results.multi_hand_landmarks:
        x_min, y_min = 1, 1
        x_max, y_max = 0, 0
        for lm in hand_landmarks.landmark:
            x_min = min(x_min, lm.x)
            y_min = min(y_min, lm.y)
            x_max = max(x_max, lm.x)
            y_max = max(y_max, lm.y)
        h, w, c = frame.shape
        x_min = int(x_min * w)
        y_min = int(y_min * h)
        x_max = int(x_max * w)
        y_max = int(y_max * h)
        color = (0, 0, 255) if label != "neutral" else (0, 255, 0)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 1)
        cv2.putText(frame, f"Action: {label}", (x_min, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
    return frame

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    result = model.predict(lm_list)
    print(f"Model prediction result: {result}")
    if result[0][0] > 0.5:
        label = "neutral"
    elif result[0][1] > 0.5:
        label = "grasping"
    elif result[0][2] > 0.5:
        label = "carrying"
    elif result[0][3] > 0.5:
        label = "cupping"
    return str(label)

def adjust_brightness(frame, brightness_factor):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,2] = hsv[:,:,2]*brightness_factor
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame

def undistort(frame, camera_matrix, dist_coeffs, dim):
    h, w = frame.shape[:2]
    K = camera_matrix
    D = dist_coeffs
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, dim, cv2.CV_16SC2)
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_frame

camera_matrix = np.array([[300.0, 0.0, 320.0],
                          [0.0, 300.0, 240.0],
                          [0.0, 0.0, 1.0]])
dist_coeffs = np.array([-0.3, 0.1, 0.0, 0.0])
dim = (640, 480) 

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 1200, 1000)

i = 0
warm_up_frames = 60

while True:
    ret, frame = cap.read()
    # cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)

    if not ret:
        break

    frame = undistort(frame, camera_matrix, dist_coeffs, dim)
    # frame = adjust_brightness(frame, 0.5)

    h, w, c = frame.shape
    crop_size = 0.8
    x_center = w // 2
    y_center = h // 2
    crop_w = int(w * crop_size)
    crop_h = int(h * crop_size)

    x1 = x_center - crop_w // 2
    x2 = x_center + crop_w // 2
    y1 = y_center - crop_h // 2
    y2 = y_center + crop_h // 2
    cropped_frame = frame[y1:y2, x1:x2]

    frame_resized = cv2.resize(cropped_frame, (w, h))

    frameRGB = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    i += 1
    if i > warm_up_frames:
        if results.multi_hand_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            if len(lm_list) == 20:
                t1 = threading.Thread(target=detect, args=(model, lm_list))
                t1.start()
                lm_list = []
            frame_resized = draw_landmark_on_image(mpDraw, results, frame_resized)
            frame_resized = draw_bounding_box_and_label(frame_resized, results, label)
        cv2.imshow("image", frame_resized)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
