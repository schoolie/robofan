# coding: utf-8
from stepper.driver import move
import time
from pi_py_darknet.darknet import initialize, detect
import cv2
from IPython.display import Image, display

from picamera import PiCamera
import time



class RoboFan(object):
    def __init__(self):
        ## Initialize darknet
        self.net, self.meta = initialize()
        self.camera = self.init_camera()


    def init_camera(self):
        ## Initialize camera
        camera = PiCamera()
        camera.rotation = 180
        camera.resolution = (640, 480)
        time.sleep(2)

        return camera


    def detect_people(self, img_filename):
        """Basic person detector. Runs yolo model on a file, filters result to only return 'person' """

        results = detect(self.net, self.meta, bytes(img_filename, 'utf-8'))

        people = []

        for cat, score, bounds in results:
            if cat == b'person':

                x, y, w, h = bounds

                center = (int(x), int(y))
                size = (w, h)
                top_left = (int(x - w / 2), int(y - h / 2))
                bottom_right = (int(x + w / 2), int(y + h / 2))
                target = (int(x), int(y-h/6))

                people.append(dict(
                    category=cat.decode("utf-8"), 
                    score=score, 
                    center=center,
                    top_left=top_left,
                    bottom_right=bottom_right,
                    target=target,
                ))

        return people


    def label(self, img_filename, people):
        """ Uses opencv to annotate image with bounding boxes and labels of detected objects """

        img = cv2.imread(img_filename)

        for n, person in enumerate(people):

            cv2.rectangle(img, person['top_left'], person['bottom_right'], (255, 0, 0), thickness=2)

            cv2.circle(img, person['target'], 3, (255,255,0)) ## target torso

            label = '{}[{}]:{:06.3f}'.format(person['category'], n, person['score']*100)
            cv2.putText(img, label, person['center'], cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0))

        return img


    def process_image(self, img_filename, result_widget=None, text_widget=None, n=0):

        start_time = time.time()

        people = self.detect_people(img_filename)

        if result_widget is not None:
            ## Label image with OpenCV and save
            img = self.label(img_filename, people)
            out_file = 'results/test_{}.jpg'.format(n)
            cv2.imwrite(out_file, img)
    #         display(Image(filename=out_file, width=640, height=480))

            file = open(out_file, "rb")
            image = file.read()
            result_widget.value = image

        ## Drive stepper
        if len(people) > 0:
            img_width = 640
            target_person = people[0]
            target_x, target_y = target_person['target']

            gain = 0.2/110 ## rough estimate of 'revolutions' per pixel
            error = abs(target_x - img_width / 2)
            pterm = error * gain

            if target_x < img_width * 0.45:
                print('Person detected at {}: moving right'.format(target_x))
                move(pterm, 1)

            elif target_x > img_width * 0.55:
                print('Person detected at {}: moving left'.format(target_x))
                move(pterm, 0)


        elapsed_time = time.time() - start_time
        print(n, len(people), '{:5.2f} seconds'.format(elapsed_time))

        return people


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

            img_filename = 'capture.jpg'

            self.camera.capture(img_filename)

            people = self.process_image(img_filename, n=n, result_widget=result_widget, text_widget=text_widget)

            n += 1


## Run script
if __name__ == '__main__':
    print('running')
    robofan = RoboFan()
    robofan.run()


