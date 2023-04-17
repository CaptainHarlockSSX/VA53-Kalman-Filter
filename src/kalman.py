import sys
import numpy as np
import cv2 as cv

def infoStartup():
	print("--- Kalman filter experiment ---")
	print("Press ESC to close the program")

def trackbarCallback(pos, _):
	thresholdValue = pos

# Find contours, process area of each contour and return the bounding rectange of the contour with the higher area
def bodyDetector(frame):
	# Store each rectangle found and its area
	areas = []
	rectangles = []

	# Get contours (vector of vectors of points)
	contoursVector, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

	if(len(contoursVector) != 0):
		for contour in contoursVector:
			if(len(contour) != 0) :
				# Process area
				area = cv.contourArea(contour)

				# Approximate the curve of the contour
				perimeter = cv.arcLength(contour, True)
				approxPoly = cv.approxPolyDP(contour, 0.02*perimeter, True)

				# Get the bounding rectange of the contour
				boundingRect = cv.boundingRect(approxPoly)

				# Apply area size condition to remove artefacts
				if(area > 1500):
					# Store rectangle and its area
					rectangles.append(boundingRect)
					areas.append(area)


	if(len(areas) != 0):
		# Get the index of the maximum area and return the corresponding Rectangle
		maxIndex = areas.index(max(areas))
		return rectangles[maxIndex]
	else:
		return 0

def draw(frame, measuredPoints, predictedPoints):
	if(len(measuredPoints) > 2):
		for i in range(len(measuredPoints)-1):
			if(measuredPoints[i] == (-1,-1) or measuredPoints[i+1] == (-1,-1)):
				pass
			else:				
				frame = cv.line(frame,measuredPoints[i],measuredPoints[i+1],(0,0,255),2)
	if(len(predictedPoints) > 2):
		for i in range(len(predictedPoints)-1):
			frame = cv.line(frame,predictedPoints[i],predictedPoints[i+1],(0,255,0),2)

def main():
	# Print usage
	infoStartup()

	# Open video
	cap = cv.VideoCapture("../Vidéo 2/1080p/Vidéo 2 60 FPS HD.mp4")

	# Create windows to display images
	cv.namedWindow("Original Frame")
	cv.namedWindow("Binary Frame")

	# Initialize keyboard input
	key = 0

	# Create a trackbar to manually determine a thresholding value for the binary frame
	cv.createTrackbar("Threshold Value","Binary Frame",0,255,trackbarCallback)
	thresholdValue = 30

	# Kernel for opening
	kernelErode = cv.getStructuringElement(cv.MORPH_RECT,(20,20))
	kernelDilate = cv.getStructuringElement(cv.MORPH_RECT,(30,30))

	# Initialize Kalman filter (Uncomment the model you need and comment others)
	dt = 1/60 # Based on video framerate

	###### Position Model ######
	# kalmanFilter = cv.KalmanFilter(2,2)
	# kalmanFilter.measurementMatrix = np.array([[1,0],[0,1]],np.float32)
	# kalmanFilter.transitionMatrix = np.array([[1,0],[0,1]],np.float32)

	###### Speed Model ######
	kalmanFilter = cv.KalmanFilter(4,2)
	kalmanFilter.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
	kalmanFilter.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],np.float32)

	###### Acceleration Model ######
	# kalmanFilter = cv.KalmanFilter(6,2)
	# kalmanFilter.measurementMatrix = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]],np.float32)
	# kalmanFilter.transitionMatrix = np.array([[1,0,dt,0,(1/2)*pow(dt,2),0],[0,1,0,dt,0,(1/2)*pow(dt,2)],[0,0,1,0,dt,0],[0,0,0,1,0,dt],[0,0,0,0,1,0],[0,0,0,0,0,1]],np.float32)

	cv.setIdentity(kalmanFilter.processNoiseCov,80)
	cv.setIdentity(kalmanFilter.measurementNoiseCov,500)

	# Initialize vectors of measured points and predicted points
	measuredPoints = []
	predictedPoints = []

	# Retain previous frame
	previousFrame = None

	# If a video device is found, start the loop to read all video frames
	if(cap.isOpened()):
		while(True):
			# Capture keayboard input
			key = cv.pollKey()

			# Read a frame
			ret, currentFrame = cap.read()

			# Exit loop if frame is empty (video is finished)
			if(np.shape(currentFrame) == ()):
				cv.imwrite("../img/1080p60fpsP.jpg",previousFrame)
				break

			# Convert frame from BGR to Gray Scale
			grayFrame = cv.cvtColor(currentFrame, cv.COLOR_BGR2GRAY)

			# Binarize image using a threshold
			# thresholdValue = int(cv.getTrackbarPos("Threshold Value","Binary Frame"))
			retVal, binaryFrame = cv.threshold(grayFrame, int(thresholdValue), 255, cv.THRESH_BINARY_INV)

			# Process an opening
			binaryFrame = cv.erode(binaryFrame, kernelErode)
			binaryFrame = cv.dilate(binaryFrame, kernelDilate)

			# Detection
			rectangle = bodyDetector(binaryFrame)
			measure = None
			if(rectangle != 0):
				currentFrame = cv.rectangle(currentFrame, rectangle, (255,0,255),4)
				x, y, w, h = rectangle[0], rectangle[1], rectangle[2], rectangle[3]
				measure = np.array([np.float32(x + w / 2), np.float32(y + h/2)])
				measuredPoints.append((int(measure[0]), int(measure[1])))

			# Launch kalman filter at the first measure
			if(len(measuredPoints) != 0):
				# Correct the kalman filter with the measure or the last prediction,
				# depending on the success of the measurement
				if(measure is None):
					measuredPoints.append((-1,-1))
				else:
					kalmanFilter.correct(measure)

				# Predict the next point
				prediction = kalmanFilter.predict()
				predictedPoints.append((int(prediction[0]),int(prediction[1])))

				# Draw all the measured points and the predicted points
				draw(currentFrame, measuredPoints, predictedPoints)

			# Exit program if "ESC" is pressed
			if(key == 27):
				break

			# Show the frame
			cv.imshow("Original Video", binaryFrame)
			cv.setWindowProperty("Original Video", cv.WND_PROP_TOPMOST, 1)

			# Save frame
			previousFrame = currentFrame

	else:
		print("Video Capture failed to open")

	output = np.savetxt("prediction.csv", predictedPoints, delimiter = ";",fmt="%d")
	output = np.savetxt("mesure.csv", measuredPoints, delimiter = ";",fmt="%d")

	cap.release()
	cv.destroyAllWindows()
	print("End Program")

if __name__ == '__main__':
    main()
