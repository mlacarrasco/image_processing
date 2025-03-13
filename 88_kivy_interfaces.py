import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.slider import Slider
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
import cv2
import numpy as np

class MedianFilterApp(App):
    def build(self):
        self.image = Image(size_hint=(1, 0.8))
        self.slider = Slider(min=1, max=20, value=1, step=2, size_hint=(1, 0.1))
        self.slider.bind(value=self.update_image)
        self.label = Label(text='Kernel Size: 1', size_hint=(1, 0.1))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.image)
        layout.add_widget(self.label)
        layout.add_widget(self.slider)

        self.original_image = cv2.imread('imagenes/noiseball.png', cv2.IMREAD_GRAYSCALE)
        self.update_image(None, 1)

        return layout

    def update_image(self, instance, value):
        value = int(value)
        if value % 2 == 0:
            value += 1  # Ensure kernel size is odd
        self.label.text = f'Kernel Size: {value}'
        filtered_image = cv2.medianBlur(self.original_image, value)
        self.display_image(filtered_image)

    def display_image(self, img):
        buf1 = cv2.flip(img, 0)
        buf = buf1.tostring()
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='luminance')
        texture.blit_buffer(buf, colorfmt='luminance', bufferfmt='ubyte')
        
        self.image.texture = texture

if __name__ == '__main__':
    MedianFilterApp().run()
