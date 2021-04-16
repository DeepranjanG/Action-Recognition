import sys
import cv2
import datetime
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.uic import loadUi
from PyQt5.QtGui import QImage, QPixmap, QPalette
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QPlainTextEdit
from datetime import datetime
import time

from main import VideoDecord
from mxnet import gluon, nd, image
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms import video
from queue import Queue



class Window(QDialog):
    def __init__(self):
        super(Window, self).__init__()
        loadUi("VideoSync.ui", self)

        p = self.palette()
        p.setColor(QPalette.Window, Qt.lightGray)
        self.setPalette(p)
        self.Notification.setText("Please fill your details !!!")

        self.WebcamButton.clicked.connect(self.STARTPlayOut)
        self.VideoButton.clicked.connect(self.STARTVideo)
        self.submitButton.clicked.connect(self.InsertRecord)

    @pyqtSlot()
    def STARTPlayOut(self):
        try:
            self.filename = QFileDialog.getOpenFileName(filter="Video (*.*)")[0]
            cap = cv2.VideoCapture(self.filename)
            for i in range(101):
	            # slowing down the loop
	            time.sleep(0.05)
	            self.ProcessBar.setValue(i)

            while(cap.isOpened()):
	            ret, frame = cap.read()

	            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
	            self.displayVideo1(frame, 1)
	            if(cv2.waitKey(25) == ord('q')):
		            break

            cap.release()
            cv2.destroyAllWindows()
        except:
            pass


    def displayVideo1(self, img, window=None):

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

    def InsertRecord(self):
        loginTime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        clientID = self.ClientId.text()
        empID = self.EmpId.text()
        empName = self.EmpName.text()
        empMobile = self.EmpMobile.text()

        if empName == "":
            self.Notification.setText("Please fill your details !!!")
        else:
            with open('Records.txt', 'a+') as f:
                f.write(loginTime + ", " + clientID + ", " + empID + ", " + empName + ", " + empMobile + '\n')
            f.close
            self.Notification.setText(f'Welcome {empName} !!!')
            self.ClearAll()



    def ClearAll(self):
        self.ClientId.clear()
        self.EmpId.clear()
        self.EmpName.clear()
        self.EmpMobile.clear()



    def STARTVideo(self):

    	try:

	        self.filename = QFileDialog.getOpenFileName(filter="Video (*.*)")[0]

	        cap = cv2.VideoCapture(self.filename)

	        vr = VideoDecord(self.filename)
	        ci = vr.detect()
	        model_name = 'slowfast_4x16_resnet50_kinetics400'
	        net = get_model(model_name, nclass=400, pretrained=True)
	        pred = net(nd.array(ci))
	        allval = []
	        classes = net.classes
	        topK = 5
	        ind = nd.topk(pred, k=topK)[0].astype('int')
	        for i in range(topK):
	            val = (classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar())
	            allval.append(val)
	        
	        for i in range(101):
	            # slowing down the loop
	            time.sleep(0.05)
	            self.ProcessBar.setValue(i)

	        fourcc = cv2.VideoWriter_fourcc(*'XVID')
	        date = datetime.datetime.now()
	        out = cv2.VideoWriter(f'data/Output_{date}.avi', fourcc, 30, (int(cap.get(3)),int(cap.get(4))))

	        font = cv2.FONT_HERSHEY_SIMPLEX
	        fontScale = 1
	        color = (255, 0, 0)
	        thickness = 2

	        while(True):
	            ret, frame = cap.read()
	            if ret == True:
	                cv2.putText(frame, allval[0][0], (20, 40) , font, fontScale,color, thickness, cv2.LINE_AA, False)
	                out.write(frame)

	                cv2.namedWindow("result",cv2.WINDOW_AUTOSIZE)
	                self.displayVideo1(frame, 1)
	                if(cv2.waitKey(25) == ord('q')):
	                    break
	            else:
	                break
	        cap.release()
	        out.release()
	        cv2.destroyAllWindows()
    	except:
    	    pass
    
    
app = QApplication(sys.argv)
window = Window()
window.show()
try:
    sys.exit(app.exec_())
except:
    print("Existing!!!")
