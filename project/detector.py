import cv2 
import torch
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name="yolov5n.pt"):
        #初始化YOLODetector类
        #加载YOLO模型
        self.model = YOLO(model_name)#通过Ultralytics的YOLO接口加载模型
        #优化的置信率阈值，减少误报
        self.confidence_threshold = 0.35#最小置信度阈值，低于此值被忽略
        #只检测车辆类别
        self.vehicle_classes = [2,3,5,7]#车辆类别的ID列表（汽车、摩托车、公交车、卡车）

        #车辆尺寸过滤参数
        self.min_width = 40 #最小宽度阈值，过滤太小的目标
        self.min_height = 35#最小高度阈值，过滤太小的
        self.max_aspect_ratio = 2.5#最大宽高比，过滤异常形状目标
        self.min_aspect_ratio = 0.3#最小宽高比，过滤异常形状目标

    def detect(self,frame):
        #用YOLO模型检测
        results = self.model(frame,conf=self.confidence_threshold,imgsz=640,iou=0.45)
        #返回检测结果，应用置信率阈值，图像尺寸，IOU阈值

        detections = []#用于保存检测到的目标
        frame_height,frame_width = frame.shape[:2]#获取视频的高度宽度

        #定义兴趣区域，只处理区域内的目标
        roi_top = int(frame_height*0.35)#从35%高度开始，忽略上面的部分
        roi_bottom = frame_height#ROI的下边界是图像的底部
        roi_left = int(frame_width*0.15)#左边界是图像宽度的15%
        roi_right = int(frame_width*0.85)#右边界是宽度的85%

        #遍历所有检测结果
        for result in results:
            boxes = result.boxes#获取检测到的所有目标框
            for box in boxes:
                #获取边界框坐标：左上角（x1，y1），右下角（x2，y2）
                x1,y1,x2,y2 = map(int,box.xyxy[0].tolist())
                #获取置信率和类别id
                confidence = float(box.conf[0])#置信率
                class_id = int(box.cls[0])#类别id
                 
                #只处理属于车辆类别的目标
                if class_id not in self.vehicle_classes:
                    continue#如果不是车辆，就跳过

                #计算边界框的尺寸和位置
                width = x2 - x1#计算宽度
                height = y2 - y1#计算高度
                center_x = (x1+x2)//2#计算目标框的水平中心
                center_y = (y1+y2)//2#计算目标框的垂直中心
                aspect_ratio = width/height if height>0 else 0#计算宽高比
                area = width * height #计算目标框的面积

                #过滤条件1：尺寸过滤-排除小的目标
                if width<self.min_width or height<self.min_height:
                    if confidence<0.6:
                        continue
                    #目标尺寸小于阈值，置信度不高，就排除

                #过滤条件2：宽高比过滤-排除异常形状目标
                if aspect_ratio>self.max_aspect_ratio or aspect_ratio<self.min_aspect_ratio:
                    #对于形状过于异常的目标过滤
                    if class_id == 3 and confidence>0.6:
                        pass#对于摩托车，置信度高的例外保留
                    else:
                        continue#排除异常的目标
                
                #过滤条件3：面积过滤-面积小的排除
                min_area = 1200#设置最小面积阈值
                if area < min_area and confidence < 0.55:
                    continue#面积太小且置信度低的排除

                #过滤条件4：区域过滤-只保留兴趣区域的目标
                if not (roi_left<=center_x<=roi_right and roi_top<=center_y<=roi_bottom):
                    if confidence < 0.7:
                        continue#目标中心不在ROI内，排除

                #过滤条件5：底部边缘检查-排除接近底部边缘的目标
                bottom_distance = frame_height - y2#计算目标框距离底部距离
                if center_y > frame_height * 0.6:#如果目标在画面下部分
                    pass#不进行底部边缘过滤
                else:
                    if area < 2000 and confidence < 0.6:
                        continue#面积小且置信度低的排除
                
                #过滤条件6：外观特征过滤-排除地面标识等
                if y2 > frame_height * 0.85 and aspect_ratio > 1.5 and height < 45:
                    #如果接近画面底部且宽高比较大但高度较小，可能是地面标识
                    if confidence < 0.7:
                        continue
                
                #过滤条件7：位置稳定性过滤-排除画面顶部的低置信度目标
                if y1 < frame_height * 0.2 and confidence < 0.65:
                    #目标在画面顶部且置信度低，排除
                    continue

                #如果通过所有过滤条件，保存检测结果
                detections.append([x1,y1,x2,y2,class_id,confidence])

            return detections#返回最终的检测结果列表