import cv2
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
import os
import configparser



mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,max_num_hands=1, min_detection_confidence=0.5, model_complexity = 0)
mp_drawing = mp.solutions.drawing_utils
zero_hands = np.concatenate([np.zeros(21*3),np.zeros(21*3)])



# function to draw mediapipe landmarks on capture frame
def draw_hand_landmarks(image, result):
	if result.multi_hand_landmarks:
		for handslms in result.multi_hand_landmarks:																	# points 																	lines
			mp_drawing.draw_landmarks(image, handslms, mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(255,255,255), thickness=1,circle_radius=1),mp_drawing.DrawingSpec(color=(255,51,255), thickness=1,circle_radius=1))





#  Normalizes the coordinates
def transform_dataframe(frame):
    data = frame[:63]
    padding = frame[63:]

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

    if perc_width < 0.1:
        return False

    scale_factor = 400.0 / width_in_image

    for (idx, val) in enumerate(x):
        x[idx] = round(scale_factor * (val - left), 2)

    for (idx, val) in enumerate(y):
        y[idx] = round(scale_factor * (val - top), 2)

    for (idx, val) in enumerate(z):
        z[idx] = round(scale_factor * val, 2)

    return True


# turns the result from the landmark detector into a numpy array of:
# -------  (2 hands)x(21 landmarks)x(cartesian triplet)  ----------
# with a final shape of:
# ---------------------- (2 hands)x(63 points)  -------------------
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




# get mediapipe results from the captured frame
def mediapipe_detection(image, model):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image.flags.writeable = False
	results = model.process(image)
	image.flags.writeable = True
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	return image, results


# allows you to start capturing and labeling data from the right
# label number if there is already data in that letter
# ex: 50 videos in A (A0-A49), new captured data will automatically be
# labeled starting at A50
def get_starting_val(gesture_dir):
	vals = [i for i in os.listdir(gesture_dir) if  i != '.DS_Store']
	return len(vals)


def removeFolder(folder_dir):
	content = os.listdir(folder_dir)
	# destroy content of folder recursively
	for item in content:
		item_dir = os.path.join(folder_dir,item)
		if os.path.isdir(item_dir):
			# if it's a folder recurse through
			removeFolder(item_dir)
		else:
			# otherwise destroy and keep on chuggin'
			os.remove(item_dir)
	# destroy current folder and return
	os.rmdir(folder_dir)
	return


def captureVideo(video_dir, gesture, video_count,frame_count,video_source,setup_check):
	print("capturing video...")
	cap = cv2.VideoCapture(video_source)
	frame_num = 0
	while cap.isOpened():
		ret,frame = cap.read()
		if ret and (frame_num < frame_count):
			image, results = mediapipe_detection(frame, hands)
			draw_hand_landmarks(image,results)
			cv2.imshow('OpenCV Feed', image)
			keypoints = extract_no_order(results)
			transformed = transform_dataframe(keypoints)
			if (not setup_check) and (not ((keypoints == zero_hands).all())) and transformed:
				frame_path = os.path.join(video_dir,"{}{}_f{}".format(gesture, video_count, frame_num))
				np.save(frame_path, keypoints)
				frame_num += 1
			elif setup_check:
				frame_num += 1
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			print("frame hit")
			break
	cap.release()
	




def captureData(gesture_dir, gesture, video_count,video_source,setup_check):
	video_dir = os.path.join(gesture_dir, gesture + str(video_count))
	if (not os.path.exists(video_dir)) and (not setup_check):
		os.mkdir(video_dir)
	print("capturing video #{}".format(video_count))
	captureVideo(video_dir, gesture, video_count,10,video_source,setup_check)




def runCaptureLoop(parent_dir, gesture, number_of_vids,video_source,setup_check):
	gesture_dir = os.path.join(parent_dir, gesture)
	if not os.path.exists(gesture_dir):
		os.mkdir(gesture_dir)
	start_val = get_starting_val(gesture_dir)
	for i in range(start_val, start_val +number_of_vids):
		captureData(gesture_dir, gesture, i, video_source,setup_check)
	cv2.destroyAllWindows()


def getDataConfig():
	parser = configparser.ConfigParser()
	parser.read('config.settings')
	pd = parser.get('data_collection', 'parent_directory')
	g = parser.get('data_collection', 'gesture')
	nv = int(parser.get('data_collection', 'number_of_vids'))
	vs = int(parser.get('data_collection', 'video_source'))
	sc = parser.getboolean('data_collection','setup_check')
	return pd,g,nv,vs,sc


parent_directory, gesture, number_of_vids,video_source,setup_check = getDataConfig()

# if main folder doesn't exist then make it
if not os.path.exists(parent_directory):
	print("making data folder...")
	os.mkdir(parent_directory)


runCaptureLoop(parent_directory, gesture, number_of_vids, video_source, setup_check)
