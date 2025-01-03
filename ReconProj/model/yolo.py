"""YOLOV8xp for keypoint detection."""
# import cv2
import torch
from ultralytics import YOLO

__all__ = ["YoloCorner"]


class YoloCorner:
    """YOLOV8xp for keypoint detection."""

    def __init__(self, weights_path: str = None):
        self.model = YOLO(weights_path)

    @torch.no_grad()
    def __call__(self, img):
        """Main func."""
        results = self.model.predict(source=img,verbose=False)
        # bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
        bboxes_keypoints = (
            results[0].keypoints.xy.cpu().numpy().astype("uint32"), #坐标信息
            results[0].boxes.cls.cpu().numpy().astype('uint32') #检测对象类别
        )
        return bboxes_keypoints
