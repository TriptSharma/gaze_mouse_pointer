import cv2
from openvino.inference_engine import IECore

class FaceDetectionModel:
    '''
    Class for the Face Detection Model.
    '''
	def __init__(self, model_name, device='CPU', extension=None, threshold=0.5):
		self.model_structure = model_name+'.xml'
		self.model_weights = model_name+'.bin'	
		self.device = device
		self.extension = extenison
        self.threshold = threshold
		
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
        face_bb = self.preprocess_output(outputs)
		return face_bb

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
        Before feeding the output of this model to the next model,
        preprocess the output to return only the bounding boxes
        '''
    	bb=[]
    	for x in outputs[0][0]:
            if x[2]>=self.threshold:
                xmin = int(x[3]*width)
                ymin = int(x[4]*height)
                xmax = int(x[5]*width)
                ymax = int(x[6]*height)
                bb.append([xmin,ymin,xmax,ymax])
        return bb
