import os
import sys
from os.path import abspath, join
root_dir = abspath(join(__file__, os.pardir, os.pardir))
sys.path.append(root_dir)

from ReconProj.track import TrafficSignTrack

if __name__ =="__main__":
    label_file_path = ""
    img_dir=""
    cache_img_dir=""
    cache_mask_dir = ""
    cache_corner_dir = ""
    mask_modelweight_path=""
    yolo_modelweight_path=""

    Tracker = TrafficSignTrack(
        label_file_path,
        img_dir,
        cache_img_dir,
        cache_mask_dir,
        cache_corner_dir,
        mask_modelweight_path,
        yolo_modelweight_path
    )

    Tracker()