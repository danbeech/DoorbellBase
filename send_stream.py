# USAGE
# python3 send_stream.py --model mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite --labels mobilenet_ssd_v2/coco_labels.txt
# [-] - for non-functional print outs to Electron
# [+] - for functional print outs to Electron

# import the necessary packages
from edgetpu.detection.engine import DetectionEngine
from imutils.video import VideoStream
from PIL import Image
import argparse
import imutils
import time
import cv2
from datetime import datetime
import numpy as np
import imagezmq
import socket


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to TensorFlow Lite object detection model")
ap.add_argument("-l", "--labels", required=True,
	help="path to labels file")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the labels dictionary
print("[-] Parsing class labels...", flush=True)
time.sleep(0.05)
labels = {}

# loop over the class labels file
for row in open(args["labels"]):
	# unpack the row and update the labels dictionary
	(classID, label) = row.strip().split(maxsplit=1)
	labels[int(classID)] = label.strip()

# load the Google Coral object detection model
print("[-] Loading Coral model...", flush=True)
time.sleep(0.05)
model = DetectionEngine(args["model"])

# initialize the ImageSender object with the socket address of the
# server
sender = imagezmq.ImageSender(connect_to="tcp://192.168.0.13:5555")

# get host name
rpiName = socket.gethostname()

# initialize the video stream and allow the camera sensor to warmup
print("[-] Starting video stream...", flush=True)
time.sleep(0.05)
vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=False).start()
time.sleep(2.0)

# init detection timer things
timeDetectedList = []
hasDetection = False

# loop over the frames from the video stream
while True:
	print('[+] Camera enabled', flush=True)
	time.sleep(0.05)
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 500 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	orig = frame.copy()

	# prepare the frame for object detection by converting (1) it
	# from BGR to RGB channel ordering and then (2) from a NumPy
	# array to PIL image format
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = Image.fromarray(frame)

	# make predictions on the input frame
	start = time.time()
	results = model.detect_with_image(frame, threshold=args["confidence"],
		keep_aspect_ratio=True, relative_coord=False)
	end = time.time()

	# if results is not empty then there is a detection
	if results:
		hasDetection = True
		# if no time of detection then it is new, set current time into dict
		if not timeDetectedList:
			timeDetectedList.append(datetime.now())
	# if there are no results then remove details of previous detection
	else:
		hasDetection = False
		timeDetectedList = []

	# if it has been 5 seconds since the person was detected, notify Electron that there is a detection
	if timeDetectedList:
		if (datetime.now() - timeDetectedList[0]).total_seconds() >= 5:
			print('[+] Person detected for 5 seconds', flush=True)
			time.sleep(0.05)

	# loop over the results
	for r in results:
		# extract the bounding box and box and predicted class label
		box = r.bounding_box.flatten().astype("int")
		(startX, startY, endX, endY) = box
		label = labels[r.label_id]

		# draw the bounding box and label on the image
		cv2.rectangle(orig, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		text = "{}: {:.2f}%".format(label, r.score * 100)
		cv2.putText(orig, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# send the output frame and wait for a key press
	sender.send_image(rpiName, orig)
	time.sleep(0.1)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
