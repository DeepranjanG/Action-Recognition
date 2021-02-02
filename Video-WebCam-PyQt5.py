import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QPalette
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QCamera
from PyQt5.QtMultimediaWidgets import QVideoWidget, QCameraViewfinder
from PyQt5.QtCore import Qt, QUrl


class Window(qtw.QWidget):
    def __init__(self):
        super().__init__()
        # Set Window Title
        self.setWindowTitle("PyQt5 Demo")
        self.setGeometry(350, 100, 700, 500)

        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)

        self.init_ui()

        #Show the app
        self.show()

    def init_ui(self):

        # create video player object
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        # create videowidget object
        videowidget = QVideoWidget()

        # Create video open button
        openBtn = qtw.QPushButton('Open Video')
        openBtn.clicked.connect(self.open_file)


        # Create webcam open button
        webcamBtn = qtw.QPushButton('Open Cam')
        webcamBtn.clicked.connect(self.open_webcam)



        # Create button for playing
        self.playBtn = qtw.QPushButton()
        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(qtw.QStyle.SP_MediaPlay))
        self.playBtn.clicked.connect(self.play_video)


        # create slider
        self.slider = qtw.QSlider(Qt.Horizontal)
        self.slider.setRange(0,0)
        self.slider.sliderMoved.connect(self.set_position)

        # Create label
        self.label = qtw.QLabel()
        self.label.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Maximum)


        # Create hbox layout
        hboxLayout = qtw.QHBoxLayout()
        hboxLayout.setContentsMargins(0,0,0,0)

        # set widgets to the hbox layout
        hboxLayout.addWidget(openBtn)
        hboxLayout.addWidget(self.playBtn)
        hboxLayout.addWidget(self.slider)



        # create vbox layout
        vboxLayout = qtw.QVBoxLayout()
        vboxLayout.addWidget(videowidget)
        vboxLayout.addLayout(hboxLayout)
        vboxLayout.addWidget(self.label)
        vboxLayout.addWidget(webcamBtn)


        self.setLayout(vboxLayout)

        self.mediaPlayer.setVideoOutput(videowidget)

        #media player signals

        self.mediaPlayer.stateChanged.connect(self.mediastate_changed)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)


    def open_file(self):
        filename, _ = qtw.QFileDialog.getOpenFileName(self, "Open Video")

        if filename != '':
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
            self.playBtn.setEnabled(True)


    def play_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()

        else:
            self.mediaPlayer.play()

    def mediastate_changed(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playBtn.setIcon(
                self.style().standardIcon((qtw.QStyle.SP_MediaPause))
            )

        else:
            self.playBtn.setIcon(
                self.style().standardIcon((qtw.QStyle.SP_MediaPlay))
            )

    def position_changed(self, position):
        self.slider.setValue(position)


    def duration_changed(self, duration):
        self.slider.setRange(0, duration)


    def set_position(self, position):
        self.mediaPlayer.setPosition(position)


    def handle_errors(self):
        self.playBtn.setEnabled(False)
        self.label.setText("Error: " + self.mediaPlayer.errorString())
        

    def open_webcam(self):

        self.exist = QCameraViewfinder()
        self.exist.show()
        self.setWindowTitle("WebCam")

        self.my_webcam = QCamera()
        self.my_webcam.setViewfinder(self.exist)
        self.my_webcam.setCaptureMode(QCamera.CaptureStillImage)
        self.my_webcam.error.connect(lambda: self.alert(self.my_webcam.errorString()))
        self.my_webcam.start()

    def alert(self, s):
        """
        This handle errors and displaying alerts.
        """
        err = qtw.QErrorMessage(self)
        err.showMessage(s)




app = qtw.QApplication([])
mw = Window()


# Run the App
app.exec_()