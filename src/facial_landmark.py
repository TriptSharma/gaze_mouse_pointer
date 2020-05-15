import cv2
impport numpy as np
from openvino.inference_engine import IECore

class FacialLandmarkEstimationModel:
	'''
	Class for the Face Landmark Model.
	'''
	def __init__(self, model_name, device='CPU', extension=None):
		self.model_structure = model_name+'.xml'
		self.model_weights = model_name+'.bin'	
		self.device = device
		self.extension = extenison
		
		self.core = IECore()
		self.net = IENetwork(model_structure, model_weights)
		self.ex = None
		
		self.input = next(iter(self.net.inputs)
		self.output = next(iter(self.net.outputs)
		self.input_shape = self.input.shape
		self.output_shape = self.output.shape

	def load_model(self):
		'''
		This method is for loading the model to the device specified by the user.
		If your model requires any Plugins, this is where you can load them.
		'''
		self.ex  = self.core.load_network(self.net, self.device)	

	def predict(self, image):
		'''
		This method is meant for running predictions on the input image.
		'''
		pp_image = self.preprocess_input(image)
		input_d = {self.input:pp_image}
		outputs = self.ex.infer(input_d)
        coords = self.preprocess_output(outputs)

		h,w = image.shape
		coords = coords * np.array([w,h,w,h])
		coords = coords.astype(np.int32)

		size = 20
		leye_bb = [coords[0]-size, coords[0]+size, coords[1]-size, coords[1]+size]
		reye_bb = [coords[2]-size, coords[2]+size, coords[3]-size, coords[3]+size]

		leye = image[leye_bb[0]:leye_bb[1], leye_bb[2]:leye_bb[3]]
		reye = image[reye_bb[0]:reye_bb[1], reye_bb[2]:reye_bb[3]]

		return leye, reye [leye_bb, reye_bb]

	def check_model(self):
		raise NotImplementedError

	def preprocess_input(self, image):
		'''
		Before feeding the data into the model for inference,
		you might have to preprocess it. This function is where you can do that.
		'''
		pp = cv2.resize(image,(self.input_shape[3],self.input_shape[2])
		pp = pp.transpose(2,0,1)
		pp = pp.reshape(1, *pp.shape)
		return pp

	def preprocess_output(self, outputs):
		'''
		Before feeding the output to gaze estimation,
		we have to preprocess the output to return left and right eye coords in range [0,1].
		'''
		output = outputs[self.output_names][0]
        leye_x = output[0].tolist()[0][0]
        leye_y = output[1].tolist()[0][0]
        reye_x = output[2].tolist()[0][0]
        reye_y = output[3].tolist()[0][0]

		return [leye_x,leye_y,reye_x,reye_y]