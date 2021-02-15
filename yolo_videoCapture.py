import sys
import datetime
from PyQt5.QtCore import pyqtSlot, Qt, QDate
from PyQt5.uic import loadUi
from PyQt5.QtGui import QImage, QPixmap, QPalette
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog

import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import core.utils as utils
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import numpy as np



class Window(QDialog):
    def __init__(self):
        super(Window, self).__init__()
        loadUi("VideoSync.ui", self)
        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)

        # Update time
        now = QDate.currentDate()
        current_date = now.toString('ddd dd MMMM yyyy')
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        self.Date_Label.setText(current_date)
        self.Time_Label.setText(current_time)

        self.WebcamButton.clicked.connect(self.STARTWebCam)
        self.VideoButton.clicked.connect(self.STARTVideo)

    @pyqtSlot()
    def STARTWebCam(self):
        cap = cv2.VideoCapture(0)

        saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
        frame_num = 0

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        date = datetime.datetime.now()
        out = cv2.VideoWriter(f'data/video/Video_{date}.avi', fourcc, 10, (640, 480))
        frame_num = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if(ret == True):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_num += 1
                image = Image.fromarray(frame)

                frame_size = frame.shape[:2]
                image_data = cv2.resize(frame, (416, 416))
                image_data = image_data / 255.
                image_data = image_data[np.newaxis, ...].astype(np.float32)
                start_time = time.time()

                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                    scores=tf.reshape(
                        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=0.45,
                    score_threshold=0.50
                )

                # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
                original_h, original_w, _ = frame.shape
                bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

                pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

                # read in all class names from config
                class_names = utils.read_class_names(cfg.YOLO.CLASSES)

                # by default allow all classes in .names file
                allowed_classes = list(class_names.values())

                # count objects found
                counted_classes = count_objects(pred_bbox, by_class=True, allowed_classes=allowed_classes)
                # loop through dict and print
                for key, value in counted_classes.items():
                    print("Number of {}s: {}".format(key, value))
                image = utils.draw_bbox(frame, pred_bbox, False, counted_classes, allowed_classes=allowed_classes,
                                        read_plate=False)

                fps = 1.0 / (time.time() - start_time)
                print("FPS: %.2f" % fps)
                result = np.asarray(image)
                result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)



                # cv2.imshow('object detection', cv2.resize(frame, (640, 480)))
                # cv2.imshow(self.displayVideo1(frame, 1), frame)
                self.displayVideo1(result, 1)
                out.write(result)
                if(cv2.waitKey(25) == ord('q')):
                    break
            else:
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()

    def displayVideo1(self, img, window=1):

        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outimg = QImage(img, img.shape[1], img.shape[0],img.strides[0], qformat)
        outimg = outimg.rgbSwapped()
        # outimg = img

        self.VideoLabel.setPixmap(QPixmap.fromImage(outimg))
        self.VideoLabel.setScaledContents(True)


    def STARTVideo(self):

        self.filename = QFileDialog.getOpenFileName(filter="Video (*.*)")[0]

        cap = cv2.VideoCapture(self.filename)

        saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
        frame_num = 0
        while(True):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_num += 1
                image = Image.fromarray(frame)
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (416, 416))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.45,
                score_threshold=0.50
            )

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

            pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())

            # count objects found
            counted_classes = count_objects(pred_bbox, by_class=True, allowed_classes=allowed_classes)
            # loop through dict and print
            for key, value in counted_classes.items():
                print("Number of {}s: {}".format(key, value))
            image = utils.draw_bbox(frame, pred_bbox, False, counted_classes, allowed_classes=allowed_classes,
                                    read_plate=False)

            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            result = np.asarray(image)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            # cv2.imshow('object detection', cv2.resize(frame, (640, 480)))
            # cv2.imshow(self.displayVideo1(frame, 1), frame)
            self.displayVideo1(result, 1)
            if(cv2.waitKey(25) == ord('q')):
                break

        cap.release()
        cv2.destroyAllWindows()


app = QApplication(sys.argv)
window = Window()
window.show()
try:
    sys.exit(app.exec_())
except:
    print("Existing!!!")