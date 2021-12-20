# Import kivy dependencies first
# import os

import cv2
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture

# from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label

import numpy as np

# import tensorflow as tf
import mediapipe as mp


# Build app and layout
class CamApp(App):
    def build(self):
        # Main layout components
        self.activate = False
        self.web_cam = Image(size_hint=(1, 0.8))
        self.button = Button(
            text="Activate or Deactivate counter of curl",
            on_press=self.active,
            size_hint=(1, 0.1),
        )
        self.lb = "Active"

        self.counter = 0
        self.stage = None

        self.verification_label = Label(text=self.lb, size_hint=(1, 0.1))
        # Mediapipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Add items to layout
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout

    # Run continuously to get webcam feed
    def update(self, *args):
        with self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as pose:
            # Read frame from opencv
            _, frame = self.capture.read()
            # frame = frame[120 : 120 + 250, 200 : 200 + 250, :]

            if self.activate:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(
                        color=(245, 117, 66), thickness=2, circle_radius=2
                    ),
                    self.mp_drawing.DrawingSpec(
                        color=(150, 7, 66), thickness=2, circle_radius=2
                    ),
                )

                try:
                    landmarks = results.pose_landmarks.landmark
                    shoulder, elbow, wrist = 11, 13, 15
                    angle = self.calculate_angle(
                        *self.conv([shoulder, elbow, wrist], landmarks)
                    )

                    if angle > 160:
                        self.stage = "down"
                    if angle < 30 and self.stage == "down":
                        self.stage = "up"
                        self.counter += 1
                except:
                    pass
                cv2.rectangle(frame, (0, 0), (225, 73), (245, 117, 16), -1)
                cv2.putText(
                    frame,
                    "Reps " + str(self.counter),
                    (10, 39),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                # Flip horizontall and convert image to texture
            buf = cv2.flip(frame, 0).tobytes()
            img_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt="bgr"
            )
            img_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
            self.web_cam.texture = img_texture

    def active(self, *args):
        self.activate = False if self.activate else True

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
            a[1] - b[1], a[0] - b[0]
        )
        angle = np.abs(radians * 180 / np.pi)
        if angle > 180:
            angle = 360 - angle
        return angle

    def conv(self, value, landmarks):
        if type(value) == int:
            a = landmarks[value]
            return [a.x, a.y]
        else:
            l = []
            for val in value:
                a = landmarks[val]
                l.append([a.x, a.y])
            return l


if __name__ == "__main__":
    CamApp().run()
