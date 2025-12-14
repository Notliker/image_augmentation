import cv2

class AugmentationBase():
    def apply(self, image):
        raise NotImplementedError