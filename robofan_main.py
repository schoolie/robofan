# coding: utf-8
from stepper.driver import StepperDriver
import time
# from pi_py_darknet.darknet import initialize, detect
import cv2
from IPython.display import Image, display

from picamera.array import PiRGBArray
from picamera import PiCamera

import os
import numpy as np
import sys
from threading import Thread
import importlib.util
import matplotlib.pyplot as plt
        
    
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True
        
        
class RoboFan(object):
    def __init__(self):
        ## Initialize darknet        
        self.init_camera()
        self.init_detector()
        


    def init_camera(self):
        ## Initialize camera

        self.stepper_driver = StepperDriver()
        
        self.resolution = (1280, 720)
        
        self.videostream = VideoStream(resolution=self.resolution,framerate=10).start()

        time.sleep(1)



    def label(self, img, people):
        """ Uses opencv to annotate image with bounding boxes and labels of detected objects """

        for n, person in enumerate(people):
            
            if person['score'] > 0.5:

                cv2.rectangle(img, person['top_left'], person['bottom_right'], (255, 0, 0), thickness=2)

                cv2.circle(img, person['target'], 3, (255,255,0)) ## target torso

                label = '{}[{}]:{:06.3f}'.format(person['category'], n, person['score']*100)
                cv2.putText(img, label, person['center'], cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0))

        return img


    def process_image(self, image, result_widget=None, text_widget=None, n=0):
        start_time = time.time()

        people = self.detect_people(image)

        if result_widget is not None:
            ## Label image with OpenCV and save
            img = self.label(image, people)
            result_widget.value = cv2.imencode('.jpg', img)[1].tostring()

        ## Drive stepper
        if len(people) > 0:
            
            best = 0
            for person in people:
                if person['score'] > best:
                    target_person = person
                    best = person['score']
                    
            
            img_width = self.image.shape[1]
            target_x, target_y = target_person['target']

            gain = 0.12/110 #0.2/110 ## rough estimate of 'revolutions' per pixel
            error = abs(target_x - img_width / 2)
            pterm = error * gain

            if target_x < img_width * 0.45:
                print('Person detected at {}: moving left'.format(target_x))
                self.stepper_driver.move(pterm, 1)

            elif target_x > img_width * 0.55:
                print('Person detected at {}: moving right'.format(target_x))
                self.stepper_driver.move(pterm, 0)


        elapsed_time = time.time() - start_time
        # print(n, len(people), '{:5.2f} seconds'.format(elapsed_time))

        return people
    

    def init_detector(self):
        ## Setup model

        MODEL_NAME = 'Sample_TFLite_model'
        GRAPH_NAME = 'detect.tflite'
        LABELMAP_NAME = 'labelmap.txt'
        self.min_conf_threshold = float(0.5)
        # resW, resH = args.resolution.split('x')
        imW, imH = int(1280), int(720)
        use_TPU = False

        # Import TensorFlow libraries
        # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
        # If using Coral Edge TPU, import the load_delegate library
        pkg = importlib.util.find_spec('tflite_runtime')
        if pkg:
            from tflite_runtime.interpreter import Interpreter
            if use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if use_TPU:
            # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
            if (GRAPH_NAME == 'detect.tflite'):
                GRAPH_NAME = 'edgetpu.tflite'       

        # Get path to current working directory
        CWD_PATH = os.getcwd()

        # Path to .tflite file, which contains the model that is used for object detection
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

        # Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if labels[0] == '???':
            del(labels[0])

        # Load the Tensorflow Lite model.
        # If using Edge TPU, use special load_delegate argument
        if use_TPU:
            interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                      experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(PATH_TO_CKPT)
        else:
            interpreter = Interpreter(model_path=PATH_TO_CKPT)

        interpreter.allocate_tensors()

        # Get model details
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5

        # Initialize frame rate calculation
        self.frame_rate_calc = 1
        self.freq = cv2.getTickFrequency()
        
        self.interpreter = interpreter
        self.labels = labels

    
    def detect_people(self, image, n=0):
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape 
        image_resized = cv2.resize(image_rgb, (self.width, self.height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
        self.interpreter.invoke()

        # Retrieve detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0] # Class index of detected objects
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0] # Confidence of detected objects
        #num = self.interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
        boxes, classes, scores

        people = []
        
        for cls, score, bounds in zip(classes, scores, boxes):
            
            cat = self.labels[int(cls)]
            
            if cat == 'person' and score > 0.6:
#             if True:
        
                ymin = int(max(1,(bounds[0] * imH)))
                xmin = int(max(1,(bounds[1] * imW)))
                ymax = int(min(imH,(bounds[2] * imH)))
                xmax = int(min(imW,(bounds[3] * imW)))
                
                center = (int((xmin+xmax)/2), int((ymin+ymax)/2))
                size = (xmax-xmin, ymax-ymin)
                top_left = (xmin, ymax)
                bottom_right = (xmax, ymin)
                target = center

                people.append(dict(
                    category=cat, 
                    score=score, 
                    center=center,
                    top_left=top_left,
                    bottom_right=bottom_right,
                    target=target,
                ))

        return people
        
        
    def test_run(self, result_widget=None, text_widget=None):

        import glob
        images = glob.glob('test_captures/*')

        n = 0
        
        for impath in images:
            
            start_time = time.time()
                
            image = cv2.imread(impath)            
            
            people = self.process_image(image, n=n, result_widget=result_widget, text_widget=text_widget)

            elapsed_time = time.time() - start_time
            print(n, len(people), 'total {:5.2f} seconds'.format(elapsed_time))
            
            n += 1

            self.rawCapture.truncate(0)
            
            self.camera.close()


    def run(self, result_widget=None, text_widget=None, max_n=None):

        n = 0
        predictions = []

        while True:
            
            start_time = time.time()
            
#             self.camera.capture(self.rawCapture, format="bgr")
            image = self.videostream.read()
#             image = self.rawCapture.array    
            image = cv2.flip(image, flipCode=-1)
            self.image = image

            people = self.process_image(image, n=n, result_widget=result_widget, text_widget=text_widget)

            elapsed_time = time.time() - start_time
            # print(n, len(people), 'total {:5.2f} seconds'.format(elapsed_time))
            
            n += 1

#             self.rawCapture.truncate(0)

            if max_n is not None:
                if n > max_n:
                    break
                    self.camera.close()

## Run script
if __name__ == '__main__':
    print('running')
    robofan = RoboFan()
    robofan.run()


