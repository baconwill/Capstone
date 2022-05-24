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


def getClasses(gesture_file):
	f = open('gesture.names', 'r')
	cn = f.read().split('\n')
	f.close()
	return cn

classNames = getClasses('gesture.names')
label_map = {label:num for num, label in enumerate(classNames)}
res_dir = "new-data"

sequence_length = 10
sequences, labels = [], []

for action in classNames:
	class_dir_r = os.path.join(res_dir, action)
	values_r = os.listdir(class_dir_r)
	for vr in values_r:
		if vr == '.DS_Store':
			continue
		window = []
		for frame_num in range(sequence_length):
			res = np.load(os.path.join(res_dir, action, vr, "{}_f{}.npy".format(vr,frame_num)))
			window.append(res)
		sequences.append(window)
		labels.append(label_map[action])


X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
actions = np.array(classNames)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10,126) ))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

res = [.7, 0.2, 0.1]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs = 5, callbacks=[tb_callback])
res = model.predict(X_test)
print(res)
model.save('AllTest')
print(actions[np.argmax(res[2])])
print(actions[np.argmax(y_test[2])])











