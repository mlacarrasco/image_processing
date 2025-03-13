import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np

class MedianFilterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Median Filter Slider")
        self.setGeometry(100, 100, 400, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.image_label = QLabel()
        self.slider_label = QLabel("Kernel Size: 1")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(2)
        self.slider.setMinimum(1)
        self.slider.setMaximum(20)
        self.slider.setSingleStep(2)
        self.slider.setValue(1)
        self.slider.valueChanged.connect(self.update_image)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.slider_label)
        layout.addWidget(self.slider)

        self.central_widget.setLayout(layout)

        self.original_image = cv2.imread('imagenes/cameraman.png', cv2.IMREAD_GRAYSCALE)
        self.update_image()

    def update_image(self):
        kernel_size = self.slider.value()
        self.slider_label.setText(f"Kernel Size: {kernel_size}")
        value = int(kernel_size)
        if value % 2 == 0:
            value += 1  # Ensure kernel size is odd
        filtered_image = cv2.medianBlur(self.original_image, value)
        self.display_image(filtered_image)

    def display_image(self, image):
        height, width = image.shape
        bytes_per_line = width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MedianFilterApp()
    window.show()
    sys.exit(app.exec_())
