# Computer-Pointer-Controller

## Introduction
Computer Pointer Controller program is used to control the movement of the mouse pointer via eye movements which are computed using Deep Learning based Computer Vision models. Initially the face of the user is identified followed by identifying the eyes and head position to compute the direction that the user is looking. Thereby, controlling the cursor movements. 

## Project Set Up and Installation

### Setup

#### Prerequisites
  - You need to install openvino successfully. <br/>
  See this [guide](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html) for installing openvino.

#### Step 1
Unzip the zip folder and change directory to the folder

#### Step 2
Initialize the virtual environment:-
```
source bin/activate
```

Install requirements:-
```
pip3 install -r requirements.txt
```
#### Step 3

Download the following models by using openVINO model downloader:-

**1. Face Detection Model**
```
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"
```
**2. Facial Landmarks Detection Model**
```
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
```
**3. Head Pose Estimation Model**
```
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"
```
**4. Gaze Estimation Model**
```
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
```

## Demo

Open a new terminal and run the following commands:-

**1. Change the directory to src directory of project repository**
```
cd <PATH_TO/mouse_pointer>/src
```
**2. Run the main.py file**
```
python3 main.py -f <Path to face detection model> \
-fl <Path to facial landmarks detection model> \
-hp <Path to head pose estimation model> \
-g <Path to gaze estimation model> \
-i <Path of input video file or enter cam for taking input video from webcam> 
-flags fd fle hpe ge
```

- To run app on GPU:-
```
python3 main.py -f <Path to face detection model> \
-fl <Path to facial landmarks detection model> \
-hp <Path to head pose estimation model> \
-g <Path to gaze estimation model> \
-i <Path of input video file or enter cam for taking input video from webcam> 
-flags fd fle hpe ge
-d GPU
```
- To run app on FPGA:-
```
python3 main.py -f <Path to face detection model> \
-fl <Path to facial landmarks detection model> \
-hp <Path to head pose estimation model> \
-g <Path to gaze estimation model> \
-i <Path of input video file or enter cam for taking input video from webcam> 
-flags fd fle hpe ge
-d HETERO:FPGA,CPU
```

- Example
```
python3 main.py -fdm /home/tript/mouse_pointer/models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -hpm /home/tript/mouse_pointer/models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 -flm /home/tript/mouse_pointer/models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 -gem /home/tript/mouse_pointer/models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 -i ../test/demo.mp4 -flags fd fld hp ge
```

## Documentation

### Documentatiob of used models

1. [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
2. [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
3. [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
4. [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

### Command Line Arguments for Running the app

Following are commanda line arguments that can use for while running the main.py file ` python main.py `:-

  1. -h                : Get the information about all the command line arguments
  2. -flm    (required) : Specify the path of Face Detection model's xml file
  3. -hpm    (required) : Specify the path of Head Pose Estimation model's xml file
  4. -gem     (required) : Specify the path of Gaze Estimation model's xml file
  5. -i     (required) : Specify the path of input video file or enter cam for taking input video from webcam
  6. -d     (optional) : Specify the target device to infer the video file on the model. Suppoerted devices are: CPU, GPU,                            FPGA (For running on FPGA used HETERO:FPGA,CPU), MYRIAD.
  7. -l     (optional) : Specify the absolute path of cpu extension if some layers of models are not supported on the device.
  9. -prob  (optional) : Specify the probability threshold for face detection model to detect the face accurately from video frame.
  8. -flags (optional) : Specify the flags from fd, fld, hp, ge if you want to visualize the output of corresponding models                           of each frame (write flags with space seperation. Ex:- -flags fd fld hp).



## Benchmarks
Benchmark results of the model.

### FP32

**Inference Time** <br/> 
![inference_time_fp32_image](images/inference_time_fp32.png "Inference Time")

**Frames per Second** <br/> 
![fps_fp32_image](images/fps_fp32.png "Frames per Second")

**Model Loading Time** <br/> 
![model_loading_time_fp32_image](images/model_loading_time_fp32.png "Model Loading Time")

### FP16

**Inference Time** <br/> 
![inference_time_fp16_image](images/inference_time_fp16.png "Inference Time")

**Frames per Second** <br/> 
![fps_fp16_image](images/fps_fp16.png "Frames per Second")

**Model Loading Time** <br/> 
![model_loading_time_fp16_image](images/model_loading_time_fp16.png "Model Loading Time")

### INT8
**Inference Time** <br/> 
![inference_time_int8_image](images/inference_time_int8.png "Inference Time")

**Frames per Second** <br/> 
![fps_int8_image](images/fps_int8.png "Frames per Second")

**Model Loading Time** <br/> 
![model_loading_time_int8_image](images/model_loading_time_int8.png "Model Loading Time")

## Results
I have run the model in 5 diffrent hardware:-
1. Intel Core i5-6500TE CPU 
2. Intel Core i5-6500TE GPU 
3. IEI Mustang F100-A10 FPGA 
4. Intel Xeon E3-1268L v5 CPU 
5. Intel Atom x7-E3950 UP2 GPU

The performances have been compared on the basis of inference time, FPS and model loading time.

Looking at the graphs we can say that FPGA took more time for inference than other devices because of time required to program the gates of fpga to make it compatible with the models. Hoever, FPGA has its own advantages such as:-
- It is re-programmable unlike other hardwares.
- It has also longer life of 5-10 years.

GPU proccesed more frames per second compared to any other hardware and specially when model precision is FP16 because GPU has severals Execution units and their instruction sets are optimized for 16bit floating point data types.

- Tests were conducted by running models with different precision levels. There is an effect of precision on accuracy. Although, model size can reduce by lowing the precision from FP32 to FP16 to INT8 and inference time becomes shorter but there is loss of info due to dreduced precision and the accuracy of model can decrease. 
- Therefore when we use FP16 then we may get lower accuracy than FP32.

## Stand Out Suggestions
- Intermediate results can be observed using the ```-flags``` argument
- Option for different inputs has been provided using the ```input```  argument
 
### Edge Cases

1. If for some reason model can not detect the face then it prints unable to detect the face and read another frame till it detects the face or user closes the window.

2. If the face pose has high roll value i.e one of the eyes cannot be seen the program prints "One or both eyes cannot be detected. Please align" till it detects both eyes or user closes the window.

2. If there are more than one face detected in the frame then model takes the first detected face for control the mouse pointer.
