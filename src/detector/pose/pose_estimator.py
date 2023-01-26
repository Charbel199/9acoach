import mediapipe as mp
from typing import List
import cv2


class PoseEstimator(object):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()

        self.LEFT_HAND = [15, 17, 19, 21]
        self.RIGHT_HAND = [16, 18, 20, 22]
        self.LEFT_FOOT = [27, 29, 31]
        self.RIGHT_FOOT = [28, 30, 32]

    @staticmethod
    def _draw_rectangle_around_points(img,
                                      points: List,
                                      color=(0, 255, 0),
                                      offset=0):
        x_min = min(p[0] for p in points) - offset
        y_min = min(p[1] for p in points) - offset
        x_max = max(p[0] for p in points) + offset
        y_max = max(p[1] for p in points) + offset
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

    def get_pose(self, image):
        """
        Image needs to be in RGB

        Args:
            image:
        """
        results = self.pose.process(image)
        return results

    def draw_pose(self, image, pose):
        h, w, c = image.shape
        if pose.pose_landmarks:
            # Draw landmarks
            self.mp_draw.draw_landmarks(image, pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            left_hand_points = []
            right_hand_points = []
            left_foot_points = []
            right_foot_points = []
            for id, lm in enumerate(pose.pose_landmarks.landmark):
                point = (int(lm.x * w), int(lm.y * h))
                if id in self.LEFT_HAND:
                    left_hand_points.append(point)
                if id in self.RIGHT_HAND:
                    right_hand_points.append(point)
                if id in self.LEFT_FOOT:
                    left_foot_points.append(point)
                if id in self.RIGHT_FOOT:
                    right_foot_points.append(point)

            # Draw Green squares around hands
            color = (0, 255, 0)
            offset = 15
            self._draw_rectangle_around_points(image,
                                               left_hand_points,
                                               color=color,
                                               offset=offset)
            self._draw_rectangle_around_points(image,
                                               right_hand_points,
                                               color=color,
                                               offset=offset)
            # Draw Purple squares around feet
            color = (255, 0, 255)
            self._draw_rectangle_around_points(image,
                                               left_foot_points,
                                               color=color,
                                               offset=offset)
            self._draw_rectangle_around_points(image,
                                               right_foot_points,
                                               color=color,
                                               offset=offset)
