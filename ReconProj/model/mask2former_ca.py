"""Created at 2023-11-17 15:52:45, authored by @GuanHeng."""
# -*- encoding: utf-8 -*-

from typing import Tuple

# import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image

__all__ = ["IDX_TO_LABEL", "SIMPLIFIED_IDX_TO_LABEL", "Mask2FormerCA"]

# 标签对应类型
CATERGORIES_MAP_CN = {
    "车道": 1,
    "人行道": 2,
    "植被": 3,
    "地形": 4,
    "杆子": 5,
    "交通标志牌": 6,
    "交通灯": 7,
    "标志线": 8,
    "车道线": 9,
    "人": 10,
    "骑行者": 11,
    "两轮车": 12,
    "自行车": 12,
    "摩托车": 12,
    "三轮车": 13,
    "汽车": 14,
    "小汽车": 14,
    "卡车": 14,
    "公交车": 14,
    "火车": 14,
    "建筑物": 15,
    "围栏": 16,
    "天空": 17,
    "路锥": 18,
    "防护柱": 19,
    "指路牌": 20,
    "斑马线": 21,
    "箭头": 22,
    "导流线": 23,
    "停止线": 24,
    "三角路标": 25,
    "限速路标": 26,
    "菱形": 27,
    "自行车路标": 28,
    "减速带": 29,
    "可跨越障碍物": 30,
    "不可跨越障碍物": 31,
    "掩码": 32,
    "其他": 33,
    "立柱": 34,
    "禁停线": 35,
    "减速让行线": 36,
    "地面文字": 37,
    "停止让行线": 38,
    "禁止通行标志": 39,
    "停车杆": 39,
    "停车锁": 39,
    "收费杆": 39,
}

CATERGORIES_EN = [
    "IG",
    "road",
    "sidewalk",
    "vegetation",
    "terrain",
    "pole",
    "traffic_sign",
    "traffic_light",
    "Sign_Line",
    "lane_marking",
    "person",
    "rider",
    "cycle",
    "tricycle",
    "car",
    "building",
    "fence",
    "sky",
    "Traffic_Cone",
    "Bollard",
    "Guide_Post",
    "Crosswalk_Line",
    "Traffic_Arrow",
    "Guide_Line",
    "Stop_Line",
    "Slow_Down_Triangle",
    "Speed_Sign",
    "Diamond",
    "Bicyclesign",
    "Speedbumps",
    "traversable_obstruction",
    "untraversable_obstruction",
    "ego_mask",
    "other",
    "parking_column",
    "no_parking_line",
    "slow_down_line",
    "road_text",
    "stop_attention_line",
    "BG",
]
LABEL_TO_IDX = {cat: idx for idx, cat in enumerate(CATERGORIES_EN)}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
SIMPLIFIED_IDX_TO_LABEL = IDX_TO_LABEL


class Mask2FormerCA:
    """Warp mask2former.

    Args:
        model_path: Path to jit traced model.
        input_size: (width, height) input image size.
    """

    label_to_idx = LABEL_TO_IDX
    idx_to_label = IDX_TO_LABEL

    def __init__(self, model_path, input_size=(1920, 1080)):
        self.model = torch.jit.load(model_path)
        # self.model.eval()
        self.device = torch.device("cuda")

        self.input_size = input_size
        self.mean = torch.tensor([123.675, 116.28, 103.53])
        self.std = torch.tensor([58.395, 57.12, 57.375])

    def _preprocess(self, image):
        h, w, _ = image.shape
        (newh, neww), scale = get_output_shape(h, w, self.input_size)
        resized_image = resize_image(image, newh, neww)
        image = torch.as_tensor(resized_image.astype("float32"))
        img_tensor = (image - self.mean) / self.std
        img_tensor = img_tensor.permute(2, 0, 1).contiguous()
        dw, dh = self.input_size
        padh = dh - newh
        padw = dw - neww
        pad = (
            int(padw / 2),
            padw - int(padw / 2),
            int(padh / 2),
            padh - int(padh / 2),
        )
        img_tensor = F.pad(img_tensor, pad, "constant", 0)
        return img_tensor, pad, (w, h)

    @torch.no_grad()
    def __call__(self, image: NDArray) -> NDArray:
        """Run model."""
        # preprocess
        # input_image, pad, raw_size = self._preprocess(image)
        image = torch.as_tensor(image.astype("float32"))
        img_tensor = (image - self.mean) / self.std
        img_tensor = img_tensor.permute(2, 0, 1).contiguous()

        input_tensor = img_tensor.unsqueeze(0).to(self.device)
        h, w = input_tensor.shape[2:]

        # with torch.no_grad():
        cls_pred_list, mask_pred_list = self.model(input_tensor)
        mask_cls_results = cls_pred_list[-1]
        mask_pred_results = mask_pred_list[-1]

        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum("bqc, bqhw->bchw", cls_score, mask_pred)
        mask = np.squeeze(
            seg_logits.argmax(dim=1).cpu().numpy().transpose(1, 2, 0) + 1
        )

        # # post processing
        # tmp_mask = mask[pad[2] : h - pad[3], pad[0] : w - pad[1]].astype(
        #     np.uint8
        # )
        # scale_mask = cv2.resize(
        #     tmp_mask, raw_size, interpolation=cv2.INTER_NEAREST
        # )

        return mask.astype(np.uint8)


def get_output_shape(
    oldh: int, oldw: int, dst_size: Tuple[int, int]
) -> Tuple[int, int]:
    """Get output shape.

    Compute the output size given input size and target short edge length.
    """
    h, w = oldh, oldw
    dst_w, dst_h = dst_size[0] * 1.0, dst_size[1] * 1.0
    scale = max(h / dst_h, w / dst_w)
    newh = int(h / scale)
    neww = int(w / scale)
    assert newh == dst_size[1] or neww == dst_size[0], "resize is not true!"
    return (newh, neww), scale


def resize_image(img, new_h, new_w, interp=Image.BILINEAR):
    """Resize Image."""
    assert len(img.shape) <= 4

    if img.dtype == np.uint8:
        if len(img.shape) > 2 and img.shape[2] == 1:
            pil_image = Image.fromarray(img[:, :, 0], mode="L")
        else:
            pil_image = Image.fromarray(img)
        pil_image = pil_image.resize((new_w, new_h), interp)
        ret = np.asarray(pil_image)
        if len(img.shape) > 2 and img.shape[2] == 1:
            ret = np.expand_dims(ret, -1)
    else:
        # PIL only supports uint8
        if any(x < 0 for x in img.strides):
            img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        shape = list(img.shape)
        shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
        img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
        _PIL_RESIZE_TO_INTERPOLATE_MODE = {
            Image.NEAREST: "nearest",
            Image.BILINEAR: "bilinear",
            Image.BICUBIC: "bicubic",
        }
        mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp]
        align_corners = None if mode == "nearest" else False
        img = F.interpolate(
            img, (new_h, new_w), mode=mode, align_corners=align_corners
        )
        shape[:2] = (new_h, new_w)
        ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

    return ret
