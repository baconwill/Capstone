import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import coremltools as ct
import configparser


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

def getClasses(gesture_file):
	f = open(gesture_file, 'r')
	cn = f.read().split('\n')
	f.close()
	return cn

classNames = getClasses("gesture_indiv.names")


actions = np.array(classNames)

def getDataSource():
	parser = configparser.ConfigParser()
	parser.read('config_indiv.settings')
	vs = int(parser.get('data_collection', 'video_source'))
	return vs

source_num = getDataSource()

def get_label(index, hand, results):
	output = None
	for idx, classification in enumerate(results.multi_handedness):
		if classification.classification[0].index == index:
			label = classification.classification[0].label
			output = label
	return output

def transform_dataframe(frame):
    data = frame[:63]
    padding = frame[63:]
    # print(data)
    x = frame[:63:3]
    y = frame[1:63:3]
    z = frame[2:63:3]

    left = min(x)
    right = max(x)
    top = min(y)

    image_width = 1080.0

    perc_width = right - left
    width_in_image = image_width * (right - left)

    if width_in_image == 0:
        return False
    # if perc_width < 0.1:
    #     return False
        
    scale_factor = 400.0 / width_in_image

    for (idx, val) in enumerate(x):
        x[idx] = round(scale_factor * (val - left), 2)

    for (idx, val) in enumerate(y):
        y[idx] = round(scale_factor * (val - top), 2)

    for (idx, val) in enumerate(z):
        z[idx] = round(scale_factor * val, 2)

    return True


def extract_no_order(result):
	if result.multi_hand_landmarks:
		if len(result.multi_hand_landmarks) == 1:
			hand = result.multi_hand_landmarks[0]
			first_hand = np.array([[res.x, res.y, res.z] for res in hand.landmark]).flatten() 
			second_hand = np.zeros(21*3)
		else:
			hand1 = result.multi_hand_landmarks[0]
			hand2 = result.multi_hand_landmarks[1]
			first_hand = np.array([[res.x, res.y, res.z] for res in hand1.landmark]).flatten() 
			second_hand = np.array([[res.x, res.y, res.z] for res in hand2.landmark]).flatten() 
	else:
		first_hand = np.zeros(21*3)
		second_hand = np.zeros(21*3)
	landmark = np.concatenate([first_hand,second_hand])
	return landmark




def mediapipe_detection(image, model):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image.flags.writeable = False
	results = model.process(image)
	image.flags.writeable = True
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	return image, results


def draw_hand_landmarks(image, result):
	landmarks = []
	if result.multi_hand_landmarks:
		for handslms in result.multi_hand_landmarks:
			for lm3 in handslms.landmark:
				landmarks.append([lm3.x, lm3.y, lm3.z])
			mp_drawing.draw_landmarks(image, handslms, mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(255,255,255), thickness=1,circle_radius=1),mp_drawing.DrawingSpec(color=(255,51,255), thickness=1,circle_radius=1))


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10,126) ))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


model.load_weights('model_indiv')



colors = [(245,117,16), (117,245,16), (16,117,245)]

sequence = []
threshold = 0.8
predictions = []


cap = cv2.VideoCapture(source_num)

valid = True
while cap.isOpened():
	ret, frame = cap.read()
	image, results = mediapipe_detection(frame, hands)
	draw_hand_landmarks(image, results)

	keypoints = extract_no_order(results)
	transform = transform_dataframe(keypoints)
	sequence.append(keypoints)

	if len(sequence) >= 12:
		sequence = sequence[-10:]
		res = model.predict(np.expand_dims(sequence, axis=0))[0]
		print(actions[np.argmax(res)])
		predictions.append(np.argmax(res))
		print(res[np.argmax(res)])
	cv2.imshow('OpenCV Feed', image)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()

