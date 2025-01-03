import os
import json
import cv2
from donkeydonkey.structure import Message

__all__ = ["ClipDataset"]


class ClipDataset(object):
    def __init__(
            self,
            imgs_folder,
            label_file
        ):
        self.images_folder = imgs_folder
        with open(label_file, "r") as f:
            lines = f.readlines()
            self.labels = [
                Message.from_json(json.loads(line.rstrip("\n"))) for line in lines
            ]
        

