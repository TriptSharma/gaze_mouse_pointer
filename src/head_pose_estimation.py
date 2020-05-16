import cv2
from openvino.inference_engine import IECore, IENetwork

class HeadPoseEstimationModel:
	'''
	Class for the Face Detection Model.
	'''
	def __init__(self, model_name, device='CPU', extension=None):
		self.model_structure = model_name+'.xml'
		self.model_weights = model_name+'.bin'	
		self.device = device
		self.extension = extension
		
		self.core = IECore()
		self.net = IENetwork(self.model_structure, self.model_weights)
		self.ex = None
		
		self.input_name = next(iter(self.net.inputs))
		self.output_name = next(iter(self.net.outputs))
		self.input_shape = self.net.inputs[self.input_name].shape
		self.output_shape = self.net.outputs[self.output_name].shape

	def load_model(self):
		'''
		This method is for loading the model to the device specified by the user.
		'''
		self.ex  = self.core.load_network(self.net, self.device)	

	def predict(self, image):
		'''
		This method is meant for running predictions on the input image.
		'''
		pp_image = self.preprocess_input(image)
		input_d = {self.input_name:pp_image}
		outputs = self.preprocess_output(self.ex.infer(input_d))
		# print(outputs)

		return outputs

	def check_model(self):
		raise NotImplementedError

	def preprocess_input(self, image):
		'''
		Preprocess image : resize and adjust dimensions 
		'''
		pp = cv2.resize(image,(self.input_shape[3],self.input_shape[2]))
		pp = pp.transpose(2,0,1)
		pp = pp.reshape(1, *pp.shape)
		return pp

	def preprocess_output(self, outputs):
		'''
		Before feeding the output of this model to Gaze Estimation,
		preprocess the output.
		'''
		output = []
		output.append(outputs['angle_y_fc'].tolist()[0][0])
		output.append(outputs['angle_p_fc'].tolist()[0][0])
		output.append(outputs['angle_r_fc'].tolist()[0][0])
		return output

