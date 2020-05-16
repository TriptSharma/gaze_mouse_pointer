import cv2
import os
import logging
import numpy as np
from face_detection import FaceDetectionModel
from facial_landmark import FacialLandmarkDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder

def build_argparser():
	#Parse command line arguments.
	parser = ArgumentParser()
	parser.add_argument("-i", "--input", required=True, type=str,
						help="Specify Path to video file or enter cam for webcam")
	parser.add_argument("-fdm", "--facedetectionmodel", required=True, type=str,
						help="Specify Path to file of Face Detection model." 
						".xml and .bin should have the same name (Specify path till the .bin and .xml name only")
	parser.add_argument("-flm", "--faciallandmarkmodel", required=True, type=str,
						help="Specify Path to Facial Landmark Detection model."
							 ".xml and .bin should have the same name (Specify path till the .bin and .xml name only")
	parser.add_argument("-hpm", "--headposemodel", required=True, type=str,
						help="Specify Path to Head Pose Estimation model."
							 ".xml and .bin should have the same name (Specify path till the .bin and .xml name only")
	parser.add_argument("-gem", "--gazeestimationmodel", required=True, type=str,
						help="Specify Path to Gaze Estimation model."
							 ".xml and .bin should have the same name (Specify path till the .bin and .xml name only")
	parser.add_argument("-l", "--cpu_extension", required=False, type=str,
						default=None,
						help="MKLDNN (CPU)-targeted custom layers."
							 "Absolute path to a shared library with the"
							 "kernels impl.")
	parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
						default=0.6,
						help="Probability threshold for model to detect the face accurately from the video frame.")
	parser.add_argument("-d", "--device", type=str, default="CPU",
						help="Specify the target device to infer on: "
							 "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
							 "will look for a suitable plugin for device "
							 "specified (CPU by default)")
	parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
						default=[],
						help="Specify the flags from fd, fld, hp, ge (Seperate each flag by space)"
							 "for see the visualization of different model outputs of each frame," 
							 "fd for Face Detection, fld for Facial Landmark Detection"
							 "hp for Head Pose Estimation, ge for Gaze Estimation.")
	return parser



def main(args):
	logger = logging.getLogger()

	inputFeeder = None
	if args.input.lower()=="cam":
		inputFeeder = InputFeeder("cam")
	else:
		if not os.path.isfile(args.input):
			logger.error("Unable to find specified video file")
			exit(1)
		inputFeeder = InputFeeder("video",args.input)
	
	modelPathDict = {'FaceDetectionModel':args.facedetectionmodel, 'FacialLandmarkDetectionModel':args.faciallandmarkmodel, 
	'GazeEstimationModel':args.gazeestimationmodel, 'HeadPoseEstimationModel':args.headposemodel}
	
	
	for fileNameKey in modelPathDict.keys():
		if not os.path.isfile(modelPathDict[fileNameKey]+'.xml'):
			logger.error("Unable to find specified "+fileNameKey+" xml file")
			exit(1)
		if not os.path.isfile(modelPathDict[fileNameKey]+'.bin'):
			logger.error("Unable to find specified "+fileNameKey+" bin file")
			exit(1)
			
	fdm = FaceDetectionModel(modelPathDict['FaceDetectionModel'], args.device, args.cpu_extension, args.prob_threshold)
	fldm = FacialLandmarkDetectionModel(modelPathDict['FacialLandmarkDetectionModel'], args.device, args.cpu_extension)
	gem = GazeEstimationModel(modelPathDict['GazeEstimationModel'], args.device, args.cpu_extension)
	hpem = HeadPoseEstimationModel(modelPathDict['HeadPoseEstimationModel'], args.device, args.cpu_extension)
	
	controller = MouseController('medium','fast')
	
	inputFeeder.load_data()
	fdm.load_model()
	fldm.load_model()
	hpem.load_model()
	gem.load_model()
	
	frame_count = 0
	for ret, frame in inputFeeder.next_batch():
		if not ret:
			print('Stream finished.')
			break
		frame_count+=1
		
		# cv2.imshow('video',frame)
		# key = cv2.waitKey(60)
		
		face_coords = fdm.predict(frame)
		if len(face_coords)==0:
			logger.error("Unable to detect the face.")
			continue
		
		face_coord = face_coords[0]
		# print(face_coords)
		face = frame[face_coord[1]:face_coord[3], face_coord[0]:face_coord[2]]

		hp_out = hpem.predict(face)
		
		left_eye, right_eye, eye_coords = fldm.predict(face)
		if len(eye_coords[0])==0 or len(eye_coords[1])==0:
			logger.error("Unable to identify the eyes.")
			continue

		new_mouse_coord, gaze_vector = gem.predict(left_eye, right_eye, hp_out)
		if new_mouse_coord==(-1,-1):
			logger.error("Couldn't identify either one or both eyes. Please align")
			continue
		
		previewFlags = args.previewFlags
		if (not len(previewFlags)==0):
			preview_frame = frame
			if 'fd' in previewFlags:
				cv2.rectangle(preview_frame, (face_coord[0], face_coord[1]), (face_coord[2], face_coord[3]), (255,0,0), 3)
				# preview_frame = face
			if 'fld' in previewFlags:
				# preview_frame = face
				cv2.rectangle(face, (eye_coords[0][0]-10, eye_coords[0][2]-10), (eye_coords[0][1]+10, eye_coords[0][3]+10), (0,255,0), 3)
				cv2.rectangle(face, (eye_coords[1][0]-10, eye_coords[1][2]-10), (eye_coords[1][1]+10, eye_coords[1][3]+10), (0,255,0), 3)
				preview_frame[face_coord[1]:face_coord[3], face_coord[0]:face_coord[2]] = face
			if 'hp' in previewFlags:
				cv2.putText(preview_frame, "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hp_out[0],hp_out[1],hp_out[2]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
			if 'ge' in previewFlags:
				x, y = face_coord[0]+eye_coords[0][0]-10+int(gaze_vector[0]*1000), face_coord[1]+eye_coords[0][3]-10+int(gaze_vector[1]*1000)
				lstart = (face_coord[0]+eye_coords[0][0]-10, face_coord[1]+eye_coords[0][3]+10)
				rstart = (face_coord[0]+eye_coords[1][0]-10, face_coord[1]+eye_coords[1][3]+10)
				cv2.line(preview_frame, lstart, (x,y), (0,0,255), 2)
				cv2.line(preview_frame, rstart, (x,y), (0,0,255), 2)
				
			cv2.imshow("visualization",cv2.resize(preview_frame,(600,500)))
			if cv2.waitKey(60)==27:
				break
		
		if frame_count%5==0:
			controller.move(new_mouse_coord[0],new_mouse_coord[1])  
	logger.error("VideoStream ended...")
	cv2.destroyAllWindows()
	inputFeeder.close()
	 
	

if __name__ == '__main__':
	# Grab command line args
	args = build_argparser().parse_args()
	main(args) 

