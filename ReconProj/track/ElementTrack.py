import os
import json
import cv2
import uuid
import numpy as np
from tqdm import tqdm
from donkeydonkey.structure import Message
from ReconProj.model import Mask2FormerCA,YoloCorner

__all__ = ["TrafficSignTrack"]


# 表示单个点的坐标信息
class CorPt:
    def __init__(self):
        self.frame_timestamp = None
        self.type_index = None
        self.type = None
        self.coordinate = []
        self.ldmk_id = None

    def to_dict(self):
        info={
            "frame_timestamp":self.frame_timestamp,
            "type_index":self.type_index,
            "type":self.type,
            "coordinate": self.coordinate,
            "ldmk_id":self.ldmk_id
        }
        return info
    
        
        


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
        self.tar_element = []
        self.img_dir = img_dir
        self.cache_img_dir = cache_img_dir
        self.cache_mask_dir = cache_mask_dir
        self.cache_corner_dir = cache_corner_dir
        self.mask_model = Mask2FormerCA(mask_modelweight_path)
        self.yolo_model = YoloCorner(yolo_modelweight_path)
        self.tar_element_type = [1]
        self.ldmk_observations =[] #每张图像的


    def __call__(self):
        
        # self.ResizeAndMask()
        # self.GenerateCorner()
        self.OpticalFlow_Features()

        print("")
        

    def ResizeAndMask(self):
        for label in tqdm(self.labels,desc="Mask: "):
            timestamp = label.meta.timestamp
            for cam in self.target_cam:
                img_path = os.path.join(self.img_dir,cam,f"{timestamp}.jpeg")
                img = cv2.imread(img_path)
                img = cv2.resize(img, self.tar_img_wh, interpolation=cv2.INTER_LINEAR)
                images_outpath = os.path.join(self.cache_img_dir,f"{timestamp}.jpeg")
                cv2.imwrite(images_outpath,img)

                ori_mask_img = self.mask_model(img)
                mask_outpath = os.path.join(self.cache_mask_dir, f"{timestamp}.png")
                cv2.imwrite(mask_outpath,ori_mask_img)

    def GenerateCorner(self):
        img_keypts_dict = dict()
        for label in tqdm(self.labels,desc="CorExtrac: "):
            timestamp = label.meta.timestamp
            img_keypts_list= img_keypts_dict.setdefault(timestamp,list())
            img_path = os.path.join(self.cache_img_dir,f"{timestamp}.jpeg")
            img = cv2.imread(img_path)
            bboxes_keypoints = self.yolo_model(img)
            
            for index, type in enumerate(bboxes_keypoints[1].tolist()):
                if type not in self.tar_element_type:
                    continue
                key_point_list = bboxes_keypoints[0].tolist()[index]
                for key_point in key_point_list:
                    newpt = CorPt()
                    newpt.frame_timestamp = timestamp
                    newpt.coordinate = key_point
                    newpt.type = type
                    newpt.type_index = index
                    img_keypts_list.append(newpt)

        ''' For Debug, visualize the points on each frame,
            and save corner extraction result
        '''         
        for timestamp, img_keypts_list in tqdm(img_keypts_dict.items(),desc="saving images with corners:"):
            img_path = os.path.join(self.cache_img_dir,f"{timestamp}.jpeg")
            img = cv2.imread(img_path)

            for point in img_keypts_list:
                coor = point.coordinate
                cv2.circle(img, coor, 5, color=(0, 255, 255))
            corner_path = os.path.join(self.cache_corner_dir,f"{timestamp}.jpeg")
            cv2.imwrite(corner_path,img)
        
        # save corner extraction result
        timestamp_keypoints = dict()
        for timestamp, img_keypts_list in img_keypts_dict.items():
            kpts_info = []
            for point in img_keypts_list:
                kpts_info.append(point.to_dict())
            timestamp_keypoints[int(timestamp)] = kpts_info

        output_path ="/data/elementtrack/img_box_keypts.json"
        with open(output_path,"w") as f:
            json.dump(timestamp_keypoints,f)
        f.close()



    def OpticalFlow_Features(self,):
        '''光流估计进行匹配'''
        lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))
        file = "/data/elementtrack/img_box_keypts.json"
        with open(file,"r") as f:
            data = json.load(f)
        frames_keypts_dict = dict()
        for timestamp, pt_list in data.items():
            pt_object_list =[]
            for pt_info in pt_list:
                pt_object = self._trans_point_object(pt_info)
                pt_object_list.append(pt_object)
            frames_keypts_dict[int(timestamp)]= pt_object_list

        allpoints_observations = list()
        for index in range(len(self.labels)):
            # 获取当前帧的关键点
            cur_timestamp = self.labels[index].meta.timestamp
            next_timestamp = self.labels[index+1].meta.timestamp
            
            for point in frames_keypts_dict[cur_timestamp]:
                ldmk_id = uuid.uuid1().hex
                point.ldmk_id = ldmk_id
                allpoints_observations.append(point)
            
            # 获取下一帧的光流结果
            cur_img_path = os.path.join(self.cache_img_dir,f"{cur_timestamp}.jpeg")
            next_img_path = os.path.join(self.cache_img_dir,f"{next_timestamp}.jpeg")
            cur_img = cv2.imread(cur_img_path,cv2.IMREAD_GRAYSCALE)
            next_img = cv2.imread(next_img_path,cv2.IMREAD_GRAYSCALE)
            coor_cur_pts = np.array([pt.coordinate for pt in frames_keypts_dict[cur_timestamp]],dtype=np.float32).reshape(-1, 1, 2)
            
            next_opticalflow_pts, opticalflow_status, err = cv2.calcOpticalFlowPyrLK(cur_img,next_img,coor_cur_pts,None,**lk_params)
            next_opticalflow_pts = np.round(next_opticalflow_pts.reshape(-1,2)).astype(int).tolist()
            # 角点跟踪状态判断
            next_corner_points_list = frames_keypts_dict[next_timestamp]
            for index, statu in enumerate(opticalflow_status.tolist()):
                if statu == 0:  # 没有追踪上
                    continue
                else: # 追踪上
                    next_pt_coor = next_opticalflow_pts[index]
                    next_pt_surround= self._get_surround_pixels(self.tar_img_wh,next_pt_coor,4)
                    for corner_pt in next_corner_points_list:
                        if corner_pt.coordinate in next_pt_surround:# 范围内
                            newpt = CorPt()
                            newpt.frame_timestamp = next_timestamp
                            newpt.coordinate = next_pt_coor
                            newpt.type = corner_pt.type
                            newpt.type_index = corner_pt.type_index
                            newpt.ldmk_id = corner_pt.ldmk_id
                            allpoints_observations.append(newpt)
                        else: # 范围外，视为新的点
                            newpt = CorPt()
                            newpt.frame_timestamp = next_timestamp
                            newpt.coordinate = next_pt_coor
                            newpt.type = frames_keypts_dict[cur_timestamp][index].type
                            newpt.type_index = frames_keypts_dict[cur_timestamp][index].type_index
                            newpt.ldmk_id =  frames_keypts_dict[cur_timestamp][index].ldmk_id
                            frames_keypts_dict[next_timestamp].append(newpt)
                            allpoints_observations.append(newpt)

        # 统计每个ldmk的观测值
        ldmk_observ = dict()
        for pt in allpoints_observations:
            ldmk_id = pt.ldmk_id
            observation_list = ldmk_observ.setdefault(ldmk_id,list())
            observation_list.append(pt.to_dict())
        
        output_path ="/data/elementtrack/ldmks_observations.json"
        with open(output_path,"w") as f:
            json.dump(ldmk_observ,f)
        f.close()



    






    def Trianglation(self,):
        """三角化"""
        print("")



    # ###############################################################################
    # #################################功能函数#######################################
    # ###############################################################################

    def _get_surround_pixels(self, image_wh,point_coor, radius:int = 1):
        [center_x, center_y] = point_coor
        [width,height] = image_wh
        surround_coors = []
        
        x_min = max(center_x - radius, 0)
        x_max = min(center_x + radius, width - 1)
        y_min = max(center_y - radius, 0)
        y_max = min(center_y + radius, height - 1)

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                if x ==center_x and y == center_y:
                    continue
                surround_coors.append([x, y])
        return surround_coors
    
    def _trans_point_object(self,point_info):
        # 变量转换成对象
        newpt = CorPt()
        newpt.frame_timestamp = int(point_info["frame_timestamp"])
        newpt.coordinate = point_info["coordinate"]
        newpt.type = point_info["type"]
        newpt.type_index = point_info["type_index"]
        newpt.ldmk_id = point_info["ldmk_id"]
        return newpt
    
    def _optical_flow_corner_visual(self,):
        """绘制光流跟踪的结果"""
        lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))
        file = "/data/elementtrack/img_box_keypts.json"
        with open(file,"r") as f:
            data = json.load(f)
        frames_keypts_dict = dict()
        for timestamp, pt_list in data.items():
            pt_object_list =[]
            for pt_info in pt_list:
                pt_object = self._trans_point_object(pt_info)
                pt_object_list.append(pt_object)
            frames_keypts_dict[int(timestamp)]= pt_object_list

        


        
        print("")
