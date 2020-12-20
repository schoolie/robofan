# coding: utf-8
# from stepper.driver import move
import time
# from pi_py_darknet.darknet import initialize, detect
import cv2
from IPython.display import Image, display

from picamera import PiCamera

import os
import numpy as np
import sys
from threading import Thread
import importlib.util
import matplotlib.pyplot as plt

        
class RoboFan(object):
    def __init__(self):
        ## Initialize darknet        
        #self.camera = self.init_camera()
        self.init_detector()
        


    def init_camera(self):
        ## Initialize camera
        
        self.resolution = (640, 480)
        
        camera = PiCamera()
        camera.rotation = 180
        camera.resolution = self.resolution
        time.sleep(2)

        return camera



    def label(self, img, people):
        """ Uses opencv to annotate image with bounding boxes and labels of detected objects """

        for n, person in enumerate(people):

            cv2.rectangle(img, person['top_left'], person['bottom_right'], (255, 0, 0), thickness=2)

            cv2.circle(img, person['target'], 3, (255,255,0)) ## target torso

            label = '{}[{}]:{:06.3f}'.format(person['category'], n, person['score']*100)
            cv2.putText(img, label, person['center'], cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0))

        return img


#     def process_image(self, img_filename, result_widget=None, text_widget=None, n=0):

#         start_time = time.time()

#         people = self.detect_people(img_filename)

#         ## Label image with OpenCV and save
#         img = self.label(img_filename, people)
#         out_file = 'predicted.jpg'.format(n)
#         cv2.imwrite(out_file, img)

#         file = open(out_file, "rb")
#         image = file.read()
            
#         if result_widget is not None:
#             result_widget.value = image

#         ## Drive stepper
#         if len(people) > 0:
#             img_width = self.resolution[0]
#             target_person = people[0]
#             target_x, target_y = target_person['target']

#             gain = 0.2/110 ## rough estimate of 'revolutions' per pixel
#             error = abs(target_x - img_width / 2)
#             pterm = error * gain

#             if target_x < img_width * 0.45:
#                 print('Person detected at {}: moving right'.format(target_x))
#                 move(pterm, 1)

#             elif target_x > img_width * 0.55:
#                 print('Person detected at {}: moving left'.format(target_x))
#                 move(pterm, 0)


#         elapsed_time = time.time() - start_time
#         print(n, len(people), '{:5.2f} seconds'.format(elapsed_time))

#         return people

    def process_image(self, image, result_widget=None, text_widget=None, n=0):
        start_time = time.time()

        people = self.detect_people(image)

        ## Label image with OpenCV and save
        img = self.label(image, people)
        out_file = 'predicted.jpg'.format(n)
#         cv2.imwrite(out_file, img)

#         file = open(out_file, "rb")
#         image = file.read()

        if result_widget is not None:
            result_widget.value = cv2.imencode('.jpg', img)[1].tostring()

        ## Drive stepper
        if len(people) > 0:
            img_width = self.width
            target_person = people[0]
            target_x, target_y = target_person['target']

            gain = 0.2/110 ## rough estimate of 'revolutions' per pixel
            error = abs(target_x - img_width / 2)
            pterm = error * gain

            if target_x < img_width * 0.45:
                print('Person detected at {}: moving right'.format(target_x))
#                 move(pterm, 1)

            elif target_x > img_width * 0.55:
                print('Person detected at {}: moving left'.format(target_x))
#                 move(pterm, 0)


        elapsed_time = time.time() - start_time
        print(n, len(people), '{:5.2f} seconds'.format(elapsed_time))

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

        

#     def detect_people(self, img_filename):
#         """Basic person detector. Runs yolo model on a file, filters result to only return 'person' """

#         results = detect(self.net, self.meta, bytes(img_filename, 'utf-8'))

#         people = []

#         for cat, score, bounds in results:
#             if cat == b'person':

#                 x, y, w, h = bounds

#                 center = (int(x), int(y))
#                 size = (w, h)
#                 top_left = (int(x - w / 2), int(y - h / 2))
#                 bottom_right = (int(x + w / 2), int(y + h / 2))
#                 target = (int(x), int(y-h/6))

#                 people.append(dict(
#                     category=cat.decode("utf-8"), 
#                     score=score, 
#                     center=center,
#                     top_left=top_left,
#                     bottom_right=bottom_right,
#                     target=target,
#                 ))

#         return people
    
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
            if cat == 'person':
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
        
    def temp(self):
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):


                object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
                if object_name == 'person':
#                 if True:
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))

                    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text


        if result_widget is not None:
            result_widget.value = cv2.imencode('.jpg', image)[1].tostring()
        
        
    def test_run(self, result_widget=None, text_widget=None):

        n = 0
        predictions = []
        file_found = True
        while file_found and n < 10:

            img_filename = 'test_captures/raw_{}.jpg'.format(n+20)

            try:
                people = self.process_image(img_filename, n=n, result_widget=result_widget, text_widget=text_widget)

            except FileNotFoundError:
                file_found = False
            n += 1


    def run(self, result_widget=None, text_widget=None):

        
        n = 0
        predictions = []

        while True:
            
            start_time = time.time()
                        
            img_filename = 'capture.jpg'

            self.camera.capture(img_filename)

            people = self.process_image(img_filename, n=n, result_widget=result_widget, text_widget=text_widget)

            elapsed_time = time.time() - start_time
            print(n, len(people), 'total {:5.2f} seconds'.format(elapsed_time))
            
            n += 1


## Run script
if __name__ == '__main__':
    print('running')
    robofan = RoboFan()
    robofan.run()


