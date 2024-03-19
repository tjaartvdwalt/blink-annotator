import cv2 as cv
import dlib
import math
import numpy as np
import os


class CascadeLoadException(Exception):
    pass


class FaceDetectException(Exception):
    pass


class EAR:
    def __init__(
        self,
        cascade_file=os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "models",
            "haarcascade_frontalface_alt.xml",
        ),
        facemarks_file=os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "models",
            "shape_predictor_68_face_landmarks.dat",
        ),
    ):
        self.cascade = self.__load_cascade(cascade_file)
        self.facemarks_model = self.__load_facemarks(facemarks_file)
        self.__frame = None
        self.__facemarks = None

    def __load_cascade(self, file):
        print(os.getcwd())
        face_cascade = cv.CascadeClassifier()
        if not face_cascade.load(file):
            raise CascadeLoadException(f"Could not load cascade file: {file}")

        return face_cascade

    def __load_facemarks(self, file):
        predictor = dlib.shape_predictor(file)
        # facemark = cv.face.createFacemarkLBF()
        # facemark.loadModel(file)

        return predictor

    def __face_detect(self, frame):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)

        faces = self.cascade.detectMultiScale(frame_gray)

        if len(faces) > 0:
            max_idx = max(((v[2] * v[3]), i) for i, v in enumerate(faces))[1]
            face = faces[max_idx]

            return face
        else:
            raise FaceDetectException("No face detected!")

    def __dist(self, p1, p2):
        return math.sqrt(((p2.x - p1.x) ** 2) + ((p2.y - p1.y) ** 2))

    def __fit(self, frame):
        if self.__frame is not frame:
            face = self.__face_detect(frame)
            print(f"face: {face}")
            # x1 = face[0]
            # y2 = face[1]
            drect = dlib.rectangle(face[0], face[1], face[0] + face[2], face[1] + face[3])
            self.__facemarks = self.facemarks_model(frame, drect)

        # print(face)
        # print(self.__facemarks)
        return self.__facemarks

    def calc(self, frame):
        facemark = self.__fit(frame)

        right_vert_1 = self.__dist(facemark.part(37), facemark.part(41))
        right_vert_2 = self.__dist(facemark.part(38), facemark.part(40))
        left_vert_1 = self.__dist(facemark.part(43), facemark.part(47))
        left_vert_2 = self.__dist(facemark.part(46), facemark.part(46))

        left_horz = self.__dist(facemark.part(36), facemark.part(39))
        right_horz = self.__dist(facemark.part(42), facemark.part(45))

        left_aspect_ratio = (left_vert_1 + left_vert_2) / (2 * left_horz)
        right_aspect_ratio = (right_vert_1 + right_vert_2) / (2 * right_horz)

        return (left_aspect_ratio, right_aspect_ratio)

    def eye_marks(self, frame):
        facemark = self.__fit(frame)

        return facemark
