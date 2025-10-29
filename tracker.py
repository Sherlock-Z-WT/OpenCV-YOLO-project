import cv2
import numpy as np
import time

class MovementTracker:
    def __init__(self, line_ratio=0.9):  # 检测线在距离底部10%处
        self.enter_count = 0  # 进入计数
        self.exit_count = 0   # 离开计数
        
        # 使用传入的line_ratio参数
        self.line_position = line_ratio  # 线的位置（视频高度的比例）
        self.line_y = None  # 检测线Y坐标
        
        # 简化但有效的跟踪机制
        self.tracked_objects = {}  # 跟踪的对象 {obj_id: {data}}
        self.object_counter = 0
        self.last_cleanup_time = time.time()#上一次请理对象的时间
        self.cleanup_interval = 3.0  # 清理间隔（秒）
        
        # 车辆类型映射
        self.vehicle_types = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        # 配置参数
        self.max_track_distance = 50  # 增大跟踪距离阈值，提高跟踪成功率
        self.position_history_length = 5  # 增加历史位置长度到5帧
        self.max_inactive_time = 1.5  # 最大不活跃时间（秒）
        
        # 存储已计数的车辆ID，用于调试和避免重复计数
        self.counted_vehicles = set()
    
    def update(self, frame, detections):
        # 初始化检测线位置
        if self.line_y is None:
            self.line_y = int(frame.shape[0] * self.line_position)
            print(f"检测线已设置在 Y={self.line_y} (距离底部10%)")
        
        # 绘制黄色检测线
        cv2.line(frame, (0, self.line_y), (frame.shape[1], self.line_y), (0, 255, 255), 3)
        cv2.putText(frame, f"DETECTION LINE - 10% from bottom", 
                   (10, self.line_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        current_time = time.time()
        
        # 定期清理已计数的车辆，避免内存溢出
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self._cleanup_tracked_objects()
            self.last_cleanup_time = current_time
        
        # 处理检测结果
        new_detections = []
        for det in detections:
            if len(det) >= 6:
                x1, y1, x2, y2 = map(int, det[0:4])
                class_id = int(det[4])
                confidence = float(det[5])
                
                # 只处理车辆类型
                if class_id in self.vehicle_types:
                    vehicle_type = self.vehicle_types[class_id]
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    new_detections.append({
                        'box': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'bottom': y2,
                        'top': y1,
                        'type': vehicle_type,
                        'confidence': confidence,
                        'class_id': class_id
                    })
        
        # 跟踪对象匹配和更新
        self._update_tracked_objects(new_detections, current_time)
        
        # 绘制对象并检查计数条件
        for obj_id, obj_data in self.tracked_objects.items():
            # 获取对象信息
            x1, y1, x2, y2 = obj_data['box']
            center_x, center_y = obj_data['center']
            vehicle_type = obj_data['type']
            confidence = obj_data['confidence']
            positions = obj_data.get('positions', [])
            crossed_line = obj_data.get('crossed_line', False)
            
            # 绘制边界框和信息
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            info_text = f"{vehicle_type} ({confidence:.2f})"
            cv2.putText(frame, info_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 核心计数逻辑：当车辆跨越检测线且未计数时
            if not crossed_line and len(positions) >= 2:
                # 检查是否有过线行为
                crossed, direction = self._check_line_crossing(positions)
                
                if crossed and obj_id not in self.counted_vehicles:
                    # 执行计数
                    if direction == 'down':  # 从上往下穿过线 = ENTER
                        self.enter_count += 1
                        cv2.putText(frame, f"+1 ENTER", (x1, y1 - 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print(f"ENTER: {vehicle_type} crossed line at {center_x},{center_y}")
                    else:  # 从下往上穿过线 = EXIT
                        self.exit_count += 1
                        cv2.putText(frame, f"+1 EXIT", (x1, y1 - 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        print(f"EXIT: {vehicle_type} crossed line at {center_x},{center_y}")
                    
                    # 标记为已穿过线
                    self.tracked_objects[obj_id]['crossed_line'] = True
                    self.counted_vehicles.add(obj_id)
        
        # 返回总计数
        total_count = self.enter_count + self.exit_count
        return total_count
    
    def _update_tracked_objects(self, new_detections, current_time):
        """更新跟踪对象，使用混合匹配策略"""
        matched_indices = set()
        new_tracked = {}
        
        # 第一阶段：尝试将新检测匹配到已有的跟踪对象
        for obj_id, obj_data in self.tracked_objects.items():
            best_match = None
            best_score = -1
            best_idx = -1
            
            # 寻找最佳匹配
            for i, det in enumerate(new_detections):
                if i in matched_indices:
                    continue
                
                # 计算IoU
                iou = self._calculate_iou(obj_data['box'], det['box'])
                
                # 计算中心点距离
                distance = np.sqrt(
                    (det['center'][0] - obj_data['center'][0])**2 + 
                    (det['center'][1] - obj_data['center'][1])**2
                )
                
                # 混合评分：IoU占60%，距离占40%
                distance_score = max(0, 1 - distance/self.max_track_distance)
                score = 0.6 * iou + 0.4 * distance_score
                
                if score > best_score and distance < self.max_track_distance:
                    best_score = score
                    best_match = det
                    best_idx = i
            
            # 如果找到匹配，更新跟踪对象
            if best_match is not None:
                matched_indices.add(best_idx)
                
                # 更新历史位置
                positions = obj_data.get('positions', [])
                positions.append(best_match['center'])
                if len(positions) > self.position_history_length:
                    positions = positions[-self.position_history_length:]
                
                # 更新对象数据
                new_tracked[obj_id] = {
                    'box': best_match['box'],
                    'center': best_match['center'],
                    'bottom': best_match['bottom'],
                    'type': best_match['type'],
                    'confidence': best_match['confidence'],
                    'crossed_line': obj_data.get('crossed_line', False),
                    'positions': positions,
                    'last_seen': current_time
                }
        
        # 第二阶段：为未匹配的检测创建新的跟踪对象
        for i, det in enumerate(new_detections):
            if i not in matched_indices:
                new_id = f"obj_{self.object_counter}"
                self.object_counter += 1
                new_tracked[new_id] = {
                    'box': det['box'],
                    'center': det['center'],
                    'bottom': det['bottom'],
                    'type': det['type'],
                    'confidence': det['confidence'],
                    'crossed_line': False,
                    'positions': [det['center']],
                    'last_seen': current_time
                }
        
        # 更新跟踪对象字典
        self.tracked_objects = new_tracked
    
    def _calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1, y1, x2, y2 = box1
        a1, b1, a2, b2 = box2
        
        # 计算交集区域
        x_left = max(x1, a1)
        y_top = max(y1, b1)
        x_right = min(x2, a2)
        y_bottom = min(y2, b2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (a2 - a1) * (b2 - b1)
        
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    
    def _check_line_crossing(self, positions):
        """检查对象是否跨越了检测线，使用更宽松的条件"""
        crossed = False
        direction = None
        
        # 检查历史位置是否有跨越检测线的行为
        for i in range(1, len(positions)):
            prev_y = positions[i-1][1]
            curr_y = positions[i][1]
            
            # 判断是否跨越检测线（考虑中心点或底部位置）
            if (prev_y < self.line_y and curr_y >= self.line_y) or \
               (prev_y > self.line_y and curr_y <= self.line_y):
                crossed = True
                # 判断方向
                direction = 'down' if curr_y > prev_y else 'up'
                break
        
        return crossed, direction
    
    def _cleanup_tracked_objects(self):
        """清理不活跃的跟踪对象和已计数的对象"""
        current_time = time.time()
        to_remove = []
        
        for obj_id, obj_data in self.tracked_objects.items():
            # 移除长时间未更新的对象
            if current_time - obj_data['last_seen'] > self.max_inactive_time:
                to_remove.append(obj_id)
            # 移除已计数且不再需要跟踪的对象
            elif obj_data.get('crossed_line', False):
                # 已计数的对象保留一段时间后再移除
                if current_time - obj_data['last_seen'] > 2.0:
                    to_remove.append(obj_id)
        
        # 执行移除
        for obj_id in to_remove:
            self.tracked_objects.pop(obj_id, None)
            # 从已计数集合中移除
            if obj_id in self.counted_vehicles:
                self.counted_vehicles.remove(obj_id)
        
        if to_remove:
            print(f"清理了 {len(to_remove)} 个不活跃或已计数的对象")