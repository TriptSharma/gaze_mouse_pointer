import math
import cv2
from openvino.inference_engine import IECore, IENetwork

class GazeEstimationModel:
	'''
	Class for the Gaze Estimation Model.
	'''
	def __init__(self, model_name, device='CPU', extension=None):
		'''
		TODO: Use this to set your instance variables.
		'''
		self.model_structure = model_name+'.xml'
		self.model_weights = model_name+'.bin'	
		self.device = device
		self.extension = extension
		
		self.core = IECore()
		self.net = IENetwork(self.model_structure, self.model_weights)
		self.ex = None
		
		self.input_name = [i for i in self.net.inputs.keys()]
		self.input_shape = self.net.inputs[self.input_name[1]].shape
		self.output_name = [i for i in self.net.outputs.keys()]

	def load_model(self):
		'''
		This method is for loading the model to the device specified by the user.
		'''
		self.ex  = self.core.load_network(self.net, self.device)	

	def predict(self, leye, reye, head_pose):
		'''
		leye and reye are left and right eye images repesctively 
		head_pose is a vector of pose angles with shape [1x3] 
		'''
		pp_leye = self.preprocess_input(leye)
		pp_reye = self.preprocess_input(reye)
		input_d = {'head_pose_angles':head_pose, 'left_eye_image':pp_leye, 'right_eye_image':pp_reye}
		outputs = self.ex.infer(input_d)
		(pos_x, pos_y), gaze_vec = self.preprocess_output(outputs, head_pose)
		return (pos_x, pos_y), gaze_vec

	def check_model(self):
		raise NotImplementedError

	def preprocess_input(self, image):
		'''
		Preprocess eye images to reshape'em to [1x3x60x60].
		'''
		pp = cv2.resize(image,(self.input_shape[3],self.input_shape[2]))
		pp = pp.transpose(2,0,1)
		pp = pp.reshape(1, *pp.shape)
		return pp

	def preprocess_output(self, outputs, head_pose):
		'''
		Get the gaze vector and correct it using the roll values of head position
		'''
		print(outputs)
		gaze_vector = outputs[self.output_name[0]].tolist()[0]
		rollValue = head_pose[2]      #angle_r_fc output from HeadPoseEstimation model
		cosValue = math.cos(rollValue * math.pi / 180.0)
		sinValue = math.sin(rollValue * math.pi / 180.0)
		
		x = gaze_vector[0] * cosValue + gaze_vector[1] * sinValue
		y = -gaze_vector[0] *  sinValue+ gaze_vector[1] * cosValue

		# x, y = gaze_vector[0], gaze_vector[1]
		return (x,y), gaze_vector
