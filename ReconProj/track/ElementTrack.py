import os
import json
import cv2
import uuid
import imageio
import numpy as np
from tqdm import tqdm
from loguru import logger
from shapely.geometry import Polygon
from donkeydonkey.structure import Message
from ReconProj.model import Mask2FormerCA,YoloCorner

__all__ = ["TrafficSignTrack"]


# 要素的类别信息
class CorElem:
    def __init__(self):
        self.id = None
        self.frameid = None
        self.type = None
        self.coordinate = []
        self.element_id =None
        self.color = None

    def to_dict(self):
        info={
            "id":self.id,
            "type":self.type,
            "coordinate": self.coordinate,
            "element_id":self.element_id
        }
        return info



class ElementTracker(object):
    def __init__(self,element: CorElem):
        self.element = element
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))
        

        self.tracked_frame_id = [] # 保存跟踪过的frame id
        self.tracked_coor_list = [] # 保存跟踪过的坐标
        self.element_observation = []

        self.tracked_frame_id.append(self.element.frameid)
        self.last_corner_frame_id = self.element.frameid
        self.tracked_coor_list.append(self.element.coordinate)
        self.element_observation.append((self.element.frameid,self.element.id))
        
        self.color = np.random.randint(0, 255, size=3).tolist()        
        self.overlap_ratio = 0.8
        self.max_overlap_num = 3
        self.track_finished =  False


    def track_element(self,next_img_idx,next_elements_list,ImagesContainer):
        # 返回状态: -1:表示退出追踪，0表示没有追踪上，1表示追踪上
        # 跟踪前先做判断
        if self.tracked_frame_id[-1] - self.last_corner_frame_id > self.max_overlap_num:
            return -1, self.element_observation
        
        # 对追踪的每张图像做判断
        coor_cur_pts =  np.array(self.tracked_coor_list[-1],dtype=np.float32).reshape(-1, 1, 2)
        pre_img = ImagesContainer[self.tracked_frame_id[-1]]
        next_img = ImagesContainer[next_img_idx]
        next_opticalflow_pts, opticalflow_status, err = cv2.calcOpticalFlowPyrLK(pre_img,next_img,coor_cur_pts,None,**self.lk_params)
        next_opticalflow_pts = np.round(next_opticalflow_pts.reshape(-1,2)).astype(int).tolist()

        overlap_element = self.element_overlap(next_opticalflow_pts,next_elements_list)
        self.tracked_frame_id.append(next_img_idx)
        
        if overlap_element is None:
            # 没有追踪上，则保存光流结果
            self.tracked_coor_list.append(next_opticalflow_pts)
            return 0, None
        else:
            # 追踪上，则使用角点结果
            self.tracked_coor_list.append(overlap_element.coordinate)
            self.last_corner_frame_id=next_img_idx
            self.element_observation.append((next_img_idx,overlap_element.id))
            return 1, overlap_element.id



    def element_overlap(self,coor_list1,elements_list):
        """目标物体重叠度"""
        if elements_list is None:
            return None    
           
        polygon1 = Polygon([
            (coor_list1[0][0], coor_list1[0][1]),
            (coor_list1[1][0], coor_list1[1][1]), 
            (coor_list1[2][0], coor_list1[2][1]), 
            (coor_list1[3][0], coor_list1[3][1])
            ]
        )
        for element in elements_list:
            coor_list2 = element.coordinate
            polygon2 = Polygon([
                (coor_list2[0][0], coor_list2[0][1]),
                (coor_list2[1][0], coor_list2[1][1]), 
                (coor_list2[2][0], coor_list2[2][1]), 
                (coor_list2[3][0], coor_list2[3][1])
                ]
            )
            overlap_area = polygon1.intersection(polygon2).area
            overlap_ratio = overlap_area / polygon2.area
            if overlap_ratio >self.overlap_ratio:
                return element
            else:
                return None
        
        return None






class TrafficSignTrack(object):
    def __init__(
        self,
        label_file,
        img_dir,
        cache_img_dir,
        cache_mask_dir,
        cache_corner_dir,
        mask_modelweight_path: str,
        yolo_modelweight_path: str
    ):
        with open(label_file, "r") as f:
            lines = f.readlines()
            self.labels = [
                Message.from_json(json.loads(line.rstrip("\n"))) for line in lines
            ]
        f.close
        self.target_cam = ["cam_front"]
        self.tar_img_wh = [960, 540]
        self.overlap_ratio =0.7
        self.tar_element = []
        self.img_dir = img_dir
        self.cache_img_dir = cache_img_dir
        self.cache_mask_dir = cache_mask_dir
        self.cache_corner_dir = cache_corner_dir
        self.mask_model_path = mask_modelweight_path
        self.yolo_model_path = yolo_modelweight_path
        self.tar_element_type = [1]

    def __call__(self):
        # 基于重叠度
        # self.ExtracteFramesCorner()
        self.FramesOpticalFlow()
        print("done")

    def ResizeAndMask(self):
        """Resize images to the target size."""
        mask_model = Mask2FormerCA(self.mask_model_path)
        for label in tqdm(self.labels,desc="Mask: "):
            timestamp = label.meta.timestamp
            for cam in self.target_cam:
                img_path = os.path.join(self.img_dir,cam,f"{timestamp}.jpeg")
                img = cv2.imread(img_path)
                img = cv2.resize(img, self.tar_img_wh, interpolation=cv2.INTER_LINEAR)
                images_outpath = os.path.join(self.cache_img_dir,f"{timestamp}.jpeg")
                cv2.imwrite(images_outpath,img)

                ori_mask_img = mask_model(img)
                mask_outpath = os.path.join(self.cache_mask_dir, f"{timestamp}.png")
                cv2.imwrite(mask_outpath,ori_mask_img)
    

    def ExtracteFramesCorner(self,):
        """Extracte corner points at each frames by YOLO."""

        YoloModel = YoloCorner(self.yolo_model_path)
        img_elements_dict = dict()

        # 逐帧提取
        for frame_index, label in enumerate(tqdm(self.labels)):
            timestamp = label.meta.timestamp
            elements_dict= img_elements_dict.setdefault(timestamp,dict())

            img_path = os.path.join(self.cache_img_dir,f"{timestamp}.jpeg")
            img = cv2.imread(img_path)
            bboxes_keypoints = YoloModel(img)
            for index, type in enumerate(bboxes_keypoints[1].tolist()):
                if type not in self.tar_element_type:
                    continue
                key_point_list = bboxes_keypoints[0].tolist()[index]
                # 排除角点的不完整识别
                if key_point_list[2]==[0,0] and  key_point_list[3]==[0,0]:
                    continue
                # 排除太小的识别结果
                if self.is_element_filter(key_point_list):
                    continue
                id = uuid.uuid1().hex
                elements_dict[id] = {
                    "id":id,
                    "type":type,
                    "frameid":frame_index,
                    "element_id": None,
                    "color": np.random.randint(0, 255, size=3).tolist(),
                    "coordinate": key_point_list
                }

        # 清理空的要素集合
        img_elements_dict = {k: v for k, v in img_elements_dict.items() if v != {}}
        output_path ="/data/elementtrack/frame_elements.json"
        with open(output_path,"w") as f:
            json.dump(img_elements_dict,f)
        f.close()
    

    def FramesOpticalFlow(self,):
        """Track all element by overlap ratio."""

        file = "/data/elementtrack/frame_elements.json"
        with open(file,"r") as f:
            frame_elements = json.load(f)
        f.close()

        frames_keypts_dict = dict()
        for timestamp, element_list in frame_elements.items():
            ele_object_list =[]
            for element_id, element_info in element_list.items():
                ele_object = self.trans_element_object(element_info)
                ele_object_list.append(ele_object)
            frames_keypts_dict[int(timestamp)]= ele_object_list


        # Only load images once
        ImagesContainer_raw = dict() 
        ImagesContainer = dict()
        for frame_index, label in enumerate(tqdm(self.labels,desc="Loading images: ")):
            timestamp = label.meta.timestamp
            img_path = os.path.join(self.cache_img_dir,f"{timestamp}.jpeg")
            ImagesContainer[frame_index] = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            ImagesContainer_raw[frame_index] = cv2.imread(img_path)



        # Start tracking
        already_tracked_element = []
        tracker_container  = []
        elemen_container = []

        # 逐帧遍历
        start = False
        for index in range(len(self.labels)-1):
            cur_img_timestamp = self.labels[index].meta.timestamp
            next_img_timestamp = self.labels[index+1].meta.timestamp

            if cur_img_timestamp == 1709880865191:
                print("")

            elements_list = frames_keypts_dict.get(cur_img_timestamp,None)
            if  elements_list is None and start is False:
                continue
            start=True

            # 要素是否要生成跟踪器？被追踪上的就不生成，只对没有追踪上ele生成追踪器
            if elements_list != None:
                for element in elements_list:
                    ele_id = element.id
                    if ele_id not in already_tracked_element:
                        already_tracked_element.append(ele_id)
                        trackobject = ElementTracker(element)
                        tracker_container.append(trackobject)
                    else:
                        continue

            # 跟踪器内的容器进行跟踪，跟踪器输入是当前帧和后一帧
            next_elements_list = frames_keypts_dict.get(next_img_timestamp,None)
            for tracker in tracker_container:
                statue, info = tracker.track_element(index+1,next_elements_list,ImagesContainer)
                if statue == 1:
                    already_tracked_element.append(info)
                elif statue == 0:
                    continue
                elif statue == -1:
                    elemen_container.append(info)
                    tracker_container.remove(tracker)
        
        print("Track element finished")
        # For debug        
        self.visualize_track_result(elemen_container,frame_elements,ImagesContainer_raw)

        return elemen_container

    # ###################################################################################
    # #################################基本功能函数#######################################
    # ###################################################################################


    def trans_element_object(self,element_info):
        newele = CorElem()
        newele.id = element_info["id"]
        newele.frameid = element_info["frameid"]
        newele.coordinate = element_info["coordinate"]
        newele.type  = element_info["type"]
        newele.element_id= element_info["element_id"]
        newele.color =  element_info["color"]
        return newele

    def is_element_filter(self,coor_list:list):
        """目标物体过滤"""
        polygon = Polygon([
            (coor_list[0][0], coor_list[0][1]),
            (coor_list[1][0], coor_list[1][1]), 
            (coor_list[2][0], coor_list[2][1]), 
            (coor_list[3][0], coor_list[3][1])
            ]
        )
        if polygon.area<40*50:
            return True
        else:
            return False


    def visualize_track_result(self,elemen_container,frame_elements,ImagesContainer_raw):
        for ele_list in elemen_container:
            if len(ele_list)<2:
                continue
            color = np.random.randint(0, 255, size=3).tolist()
            for img_ele in ele_list:
                (img_idx,ele_id) = img_ele
                timestamp = self.labels[img_idx].meta.timestamp
                elel_obj_coor = frame_elements[str(timestamp)][ele_id]["coordinate"]
                
                image = ImagesContainer_raw[img_idx]
                pts = np.array(elel_obj_coor, np.int32)
                pts = pts.reshape((-1, 1, 2))
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                image[mask == 255] = np.array(color)  # 设置为目标颜色
                ImagesContainer_raw[img_idx] = image
        
        video_name = '/data/track_result.mp4'
        fps = 30
        with imageio.get_writer(video_name, fps=fps) as video:
            for index, label in enumerate(self.labels):
                timestamp =label.meta.timestamp
                img =ImagesContainer_raw[index]
                video.append_data(img)
        
        logger.info(f"save track video done.")
