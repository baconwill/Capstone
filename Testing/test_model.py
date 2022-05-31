import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import seaborn as sns
import tensorflow_addons as tfa



def getClasses(gesture_file):
	f = open('test_gesture.names', 'r')
	cn = f.read().split('\n')
	f.close()
	return cn

classNames = getClasses('test_gesture.names')
label_map = {label:num for num, label in enumerate(classNames)}
res_dir = "test_data"

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


print(np.array(sequences).shape)
actions = np.array(classNames)
X = np.array(sequences)
y = to_categorical(labels).astype(int)



model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10,126) ))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights('AllTest2')

results = model.evaluate(X, y)
predictions = model.predict(X)

ypred = np.argmax(predictions, axis=1)
ypred_lab = actions[ypred]
ymax = np.argmax(y, axis=1)
y_reshape = ymax.reshape((26,20))
y_pred_reshape = ypred.reshape((26,20))
cmat = tf.math.confusion_matrix(labels=ymax, predictions=ypred).numpy()

plt.figure(figsize = (10,10))
ax = sns.heatmap(cmat, annot=True , cmap='Blues',annot_kws={"size": 8},square=False)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual');

ax.xaxis.set_ticklabels(actions)
ax.yaxis.set_ticklabels(actions)

plt.savefig('confusion_matrix2.pdf')


metric = tfa.metrics.F1Score(num_classes=26)
metric.update_state(y, predictions)
res = metric.result()
print(res)

