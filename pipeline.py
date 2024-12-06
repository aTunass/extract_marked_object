import torch
import cv2
import time
import os
import asyncio
import shutil
import numpy as np
from PIL import Image
import math
from torchvision import transforms
from ultralytics import YOLO
from clipseg.models.clipseg import CLIPDensePredT
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class DetectedMarked:
    def __init__(self, sam_path="checkpoints/sam2.1_hiera_tiny.pt", sam_cfg="configs/sam2.1/sam2.1_hiera_t.yaml", 
                 clipseg_path="weight/weights_40.pth", yolo_path="weight/best_crop", device=None):
        
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model_sam2 = SAM2ImagePredictor(build_sam2(sam_cfg, sam_path, device=self.device), max_hole_area=20, max_sprinkle_area=20)
        print("Load model SAM2 successfully")
        
        self.model_clipseg = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
        self.model_clipseg.eval()
        self.model_clipseg.load_state_dict(torch.load(clipseg_path, map_location=self.device), strict=False)
        
        self.transform_clipseg = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((352, 352)),
        ])
        
        self.input_image_shape_detection = None
        self.prompts = ['annotated']
        self.cls_name = ["arrow", "circle", "dot", "x", "line", "other", "tick"]
        self.model_yolo = YOLO(yolo_path).to(self.device)
        self.model_yolo_det = YOLO("weight_yolo/best.pt").to(self.device)
        
    def inference_pipeline(self, image:Image.Image, is_multimask=False):
        image_np = np.array(image.convert("RGB"))
        self.image_np = image_np
        self.input_image_shape_detection = image_np.shape
        st = time.time()
        lst_bbox_marked, mask = self.inference_clipseg(image)
        print("clipseg time:", time.time()-st)     
        
        st = time.time()
        lst_bbox_yolo = self.inference_detection(image=image_np, conf=0.3)
        print("yolo time:", time.time()-st)     
        
        lst_bbox_filter = self._filter_bounding_boxes(lst_bbox_marked, lst_bbox_yolo, 0.0)   
        
        if lst_bbox_filter==[]:
            return [], None, []
        
        points = []
        marked_area_lst = []
        for i, bb in enumerate(lst_bbox_filter):
            if mask.shape[1:]!=image_np.shape[:-1]:
                mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            image_crop, bb_padding = self._crop_image_by_bbox(image=image_np, bbox=bb)
            cls_id, b_mask, conf = self.inference_classification(image=image_crop, conf=0.2)
            
            if np.count_nonzero(b_mask)>0:
                marked_area_lst.append(np.count_nonzero(b_mask))
            else:
                marked_area_lst.append(150)

            # if b_mask is not None:
                # cv2.imwrite("mask_yolo.jpg", b_mask)
            if cls_id is None:
                # print("none")
                points.append(self._find_closest_non_mask_point(mask=mask, bbox=bb, is_crop=True))
            elif cls_id==1:
                # print("circle")
                points.append(self._get_bbox_center_xyxy(bb_padding))
            elif cls_id==0:
                # print("arrow")
                points.append(self._calculate_arrow_orientation(mask_crop=b_mask, bbox=bb_padding))
            else:
                # print("mask")
                points.append(self._find_closest_non_mask_point(mask=b_mask, bbox=bb_padding, is_crop=False))
        st = time.time()
        masks = self.inference_sam2(image=image, point_input=points, is_multimask=is_multimask, marked_area_lst=marked_area_lst)
        print("inference_sam2 time:", time.time()-st)
        
        return lst_bbox_filter, masks, points
    
    def inference_clipseg(self, image: Image.Image):
        image = image.convert("RGB")
        image_transform = self.transform_clipseg(image).unsqueeze(0)
        with torch.no_grad():
            preds = self.model_clipseg(image_transform, self.prompts)[0]
        mask = torch.sigmoid(preds[0][0]).numpy()
        mask[mask < 0.05] = 0                      # Loại bỏ các pixel có confidence <= 0.2
        mask[mask >= 0.05] = 1                       
        mask = (mask * 255).astype('uint8')
        lst_bb = self._find_large_areas(mask, area_percent=0.0001)
        lst_bb = self._convert_box_xywh_to_xyxy(lst_bb, box_format="xywh")
        lst_bb = [self._map_bounding_box_to_original(box_xyxy=b, old_size=image.size, new_size=(352, 352)) for b in lst_bb]
        return self._merge_overlapping_bboxes(lst_bb, scale=1.1), mask
    
    def inference_classification(self, image, conf=0.1):
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image
        results = self.model_yolo.predict(img, task="segment", conf=conf, device=torch.device(self.device), verbose=False)
        b_mask = np.zeros(img.shape[:2], np.uint8)
        for _, r in enumerate(results):
            cls_id = r.boxes.cls
            conf = r.boxes.conf
            # print("conf", r.boxes.conf)
            for ci, c in enumerate(r):
                b_mask = np.zeros(img.shape[:2], np.uint8)
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
        if cls_id.numel() == 0:
            return None, b_mask, None
        else:
            return cls_id[0], b_mask, conf[0]
    
    def inference_detection(self, image, conf=0.1):
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image
        results = self.model_yolo_det.predict(img, task="segment", conf=conf, device=torch.device(self.device), verbose=False)
        for _, r in enumerate(results):
            lst_bb = r.boxes.xyxy
            lst_conf = r.boxes.conf
        return self._merge_overlapping_bboxes(lst_bb.cpu().tolist(), scale=0.8)
    
    def inference_sam2(self, image: Image.Image, bbox_input=None, point_input=None, is_multimask=False, marked_area_lst=[]):
        image = np.array(image.convert("RGB"))
        
        if bbox_input is not None:
            input_box = np.array(bbox_input) 
            input_box = input_box[None, :]
        else:
            input_box = None
        
        if point_input is not None:
            num_marked = len(point_input)
            input_point = np.array(point_input).reshape(num_marked, 1, 2)
            input_label = np.array([1]*num_marked).reshape(num_marked, 1)
        else:
            input_point = None
            input_label = None
            
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.model_sam2.set_image(image)
            masks, scores, _ = self.model_sam2.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=is_multimask,
            )
        # print("scores", scores)
        # print(marked_area_lst)
        if is_multimask:
            mask_filter = []
            # print(masks.shape)
            
            if len(masks.shape)==3:
                masks = np.expand_dims(masks, axis=0)
                scores = np.expand_dims(scores, axis=0)
            
            for id, mask in enumerate(masks):
                
                scores_filter_idex = np.where(scores[id] > 0.7)[0]
                if scores_filter_idex.size > 0:
                    max_index = scores_filter_idex[np.argmax(scores[id][scores_filter_idex])]
                    mask_filter.append(np.expand_dims(mask[max_index], axis=0))
                elif marked_area_lst[id]==0:
                    max_index = np.argmax(scores[id])
                    mask_filter.append(np.expand_dims(mask[max_index], axis=0))
                else:
                    min_area = float('inf')
                    mask_save = None
                    for i, m in enumerate(mask):
                        m_area = np.count_nonzero(m)
                        # print(m_area)
                        if m_area>marked_area_lst[id]*5 and m_area<min_area:
                            mask_save = m
                            min_area=m_area
                    
                    if mask_save is None:        
                        for i, m in enumerate(mask):
                            m_area = np.count_nonzero(m)
                            if m_area>marked_area_lst[id]//3 and m_area<min_area:
                                mask_save = m
                                min_area=m_area
                    if mask_save is None:
                        mask_save = mask[0]
                            
                    mask_filter.append(np.expand_dims(mask_save, axis=0))
            return np.concatenate(mask_filter, axis=0)
        return masks
    
    def _filter_bounding_boxes(self, list1, list2, iou_threshold=0):
        result = []
        
        for box1 in list1:
            is_saved = False
            for box2 in list2:
                iou = self._compute_iou(box1, box2)
                if iou > iou_threshold:
                    result.append(box2) 
                    is_saved = True

            if not is_saved:
                result.append(box1)

        return result
    
    def _crop_image_by_bbox(self, image, bbox, padding=10):
        x_min, y_min, x_max, y_max = map(int, bbox)

        x_min = max(0, int(x_min - padding))
        y_min = max(0, int(y_min - padding))
        x_max = min(image.shape[1], int(x_max + padding))
        y_max = min(image.shape[0], int(y_max + padding))

        cropped_image = image[y_min:y_max, x_min:x_max]
        new_bb = [x_min, y_min, x_max, y_max]

        return cropped_image, new_bb
    
    def _find_closest_non_mask_point(self, mask, bbox, is_crop=False):
        x_min, y_min, x_max, y_max = map(int, bbox)
        if is_crop:
            mask_crop = mask[y_min:y_max, x_min:x_max]
        else:
            mask_crop = mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_crop = cv2.dilate(mask_crop, kernel, iterations=2)
        non_mask_indices = np.argwhere(mask_crop == 0)
        
        if non_mask_indices.size == 0:
            return (x_min-5, y_min-5)

        center_x = (x_max - x_min) / 2
        center_y = (y_max - y_min) / 2

        distances = np.sqrt((non_mask_indices[:, 1] - center_x) ** 2 + 
                            (non_mask_indices[:, 0] - center_y) ** 2)
        
        valid_indices = np.where(distances > 10)[0]
        if valid_indices.size > 0:
            closest_index = valid_indices[np.argmin(distances[valid_indices])]
        else:
            return (x_min-5, y_min-5)
        
        # closest_index = np.argmin(distances)

        y, x = non_mask_indices[closest_index]
        
        # image_test = self.image_np[y_min:y_max, x_min:x_max]
        # result, _=apply_mask_np(image_test, mask_crop)
        # cv2.circle(result, (int(x), int(y)), radius=4, color=(0, 0, 255), thickness=-1)
        # cv2.imwrite("result.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        return (x + x_min, y + y_min)
    
    def _calculate_arrow_orientation(self, mask_crop, bbox):
        x_min, y_min, x_max, y_max = map(int, bbox)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_crop = cv2.dilate(mask_crop, kernel, iterations=2)
        contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return "Không tìm thấy bất kỳ hình dạng nào trong ảnh."
    
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x, center_y = 0, 0
        if M["mu20"] != M["mu02"]:
            angle = 0.5 * math.atan2(2 * M["mu11"], M["mu20"] - M["mu02"])
            angle_deg = math.degrees(angle)  
        else:
            angle_deg = 0

        arrow_length = math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)//2+15
        width, height = mask_crop.shape[1], mask_crop.shape[0]
        
        normal_angle_rad = math.radians(angle_deg) 
        normal_vector = (math.cos(normal_angle_rad), math.sin(normal_angle_rad))

        mask1 = np.zeros_like(mask_crop, dtype=np.uint8)
        mask2 = np.zeros_like(mask_crop, dtype=np.uint8)
        
        w = x_max - x_min
        h = y_max - y_min
        for row in range(h):
            for col in range(w):
                dx = col - w//2
                dy = row - h//2

                dot_product = dx * normal_vector[0] + dy * normal_vector[1]
                if dot_product < 0:
                    mask1[row, col] = mask_crop[row, col]
                else:
                    mask2[row, col] = mask_crop[row, col]
        mask1_count, mask2_count = np.count_nonzero(mask1), np.count_nonzero(mask2)
        if mask1_count>mask2_count:
            mask_check = mask1
        else:
            mask_check = mask2
        # cv2.imwrite("mask_check.jpg", mask_check)
        left_half, right_half = mask_check[:, :width // 2], mask_check[:, width // 2:]
        # cv2.imwrite("left_half.jpg", left_half)
        # cv2.imwrite("right_half.jpg", right_half)
        left_count, right_count = np.count_nonzero(left_half), np.count_nonzero(right_half)
        top_half, bottom_half = mask_check[:height // 2, :], mask_check[height // 2:, :]
        top_count, bottom_count = np.count_nonzero(top_half), np.count_nonzero(bottom_half)
        
        if -90 < angle_deg < -45 or 45 < angle_deg < 90:
            if top_count > bottom_count:
                # print("top")
                angle_rad = math.radians(angle_deg if angle_deg < 0 else angle_deg + 180)
            else:
                # print("bottom")
                angle_rad = math.radians(angle_deg if angle_deg > 0 else angle_deg + 180)
        else:
            if right_count > left_count:
                # print("right")
                angle_rad = math.radians(angle_deg)
            else:
                # print("left")
                angle_rad = math.radians(angle_deg + 180)
               
        new_x = center_x + arrow_length * math.cos(angle_rad)
        new_y = center_y + arrow_length * math.sin(angle_rad)
        
        # output_img = cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR)  # Chuyển sang ảnh màu để vẽ
        # cv2.drawContours(output_img, [largest_contour], -1, (0, 255, 0), 2)  # Vẽ đường viền
        # cv2.circle(output_img, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)  # Đánh dấu trọng tâm ban đầu
        # cv2.circle(output_img, (int(new_x), int(new_y)), 5, (255, 0, 0), -1)  # Đánh dấu trọng tâm mới
        # cv2.arrowedLine(output_img, (int(center_x), int(center_y)), (int(new_x), int(new_y)), (255, 255, 0), 2)  # Hướng dịch chuyển
        # cv2.imwrite("output_with_moved_center.jpg", output_img)
        new_x = new_x + x_min
        new_y = new_y + y_min

        return (new_x, new_y)
    
    def _find_overlap_box(self, box1_lst, box2_lst, scale=1.2):
        overlaps = set()  
        for box1 in box1_lst:
            for box2 in box2_lst:
                if self._is_overlap(box1, box2, scale=scale):
                    overlaps.add(tuple(box2))  
        
        return list(overlaps) 
    
    def _get_bbox_center_xyxy(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        return [center_x, center_y]
    
    def _map_bounding_box_to_original(self, box_xyxy, old_size, new_size):
        original_width, original_height = old_size
        resized_width, resized_height = new_size
        scale_x = original_width / resized_width
        scale_y = original_height / resized_height
        x_min, y_min, x_max, y_max = box_xyxy
        return [
            x_min * scale_x,
            y_min * scale_y,
            x_max * scale_x,
            y_max * scale_y
        ]
    
    def _find_large_areas(self, mask, area_percent):
        min_area = int(area_percent * mask.shape[0] * mask.shape[1]) 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lst_bb = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                lst_bb.append(cv2.boundingRect(contour))

        return lst_bb
    
    def _expand_bbox(self, bbox, scale_factor=1.5, target_width=None, target_height=None):
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin

        new_width = bbox_width * scale_factor
        new_height = bbox_height * scale_factor

        new_xmin = int(center_x - new_width / 2)
        new_ymin = int(center_y - new_height / 2)
        new_xmax = int(center_x + new_width / 2)
        new_ymax = int(center_y + new_height / 2)

        if target_width and target_height:
            new_xmin = max(0, new_xmin)
            new_ymin = max(0, new_ymin)
            new_xmax = min(target_width, new_xmax)
            new_ymax = min(target_height, new_ymax)

        return new_xmin, new_ymin, new_xmax, new_ymax
    
    def _merge_overlapping_bboxes(self, bboxes, scale=1):
        merged_bboxes = []
        while bboxes:
            current_box = bboxes.pop(0)
            overlap_found = False

            for i, other_box in enumerate(bboxes):
                if self._is_overlap(current_box, other_box, scale=scale):
                    current_box = self._merge_boxes(current_box, other_box)
                    bboxes.pop(i)
                    overlap_found = True
                    break

            if overlap_found:
                bboxes.insert(0, current_box)
            else:
                merged_bboxes.append(current_box)

        return merged_bboxes
    
    def _is_overlap(self, box1, box2, scale=1.1):
        if self.input_image_shape_detection is not None:
            x1_1, y1_1, x2_1, y2_1 = self._expand_bbox(bbox=box1, scale_factor=scale, target_width=self.input_image_shape_detection[1], 
                                                       target_height=self.input_image_shape_detection[0])
            x1_2, y1_2, x2_2, y2_2 = self._expand_bbox(bbox=box2, scale_factor=scale, target_width=self.input_image_shape_detection[1], 
                                                       target_height=self.input_image_shape_detection[0])
        else:
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)

    def _merge_boxes(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        x1 = min(x1_1, x1_2)
        y1 = min(y1_1, y1_2)
        x2 = max(x2_1, x2_2)
        y2 = max(y2_1, y2_2)
        return (x1, y1, x2, y2)
        
    def _convert_box_xywh_to_xyxy(self, box, box_format="xywh"):
        if box_format=="xyxy":
            return box
        if len(box) == 4 and not isinstance(box[0], (list, tuple)):
            return [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        else:
            result = []
            for b in box:
                b = self._convert_box_xywh_to_xyxy(b)
                result.append(b)               
        return result

    def _compute_iou(self, box1, box2):
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        iou = inter_area / (box1_area + box2_area - inter_area)
        return iou

    def _calculate_bbox_area(self, bbox):
        x_min, y_min, x_max, y_max = map(int, bbox)
        width = max(0, x_max - x_min)
        height = max(0, y_max - y_min)
        
        area = width * height
        return area

    def _non_maximum_suppression_for_yolov10(self, boxes, scores, labels, iou_threshold=0.5):
        sorted_indices = np.argsort(scores)[::-1]
        selected_boxes = []
        selected_scores = []
        selected_labels = []
        
        while len(sorted_indices) > 0:
            current = sorted_indices[0]
            selected_boxes.append(boxes[current])
            selected_scores.append(scores[current])
            selected_labels.append(labels[current])
            
            remaining_indices = []
            for i in sorted_indices[1:]:
                if labels[i] == labels[current]:  
                    iou = self._compute_iou(boxes[current], boxes[i])
                    if iou < iou_threshold:
                        remaining_indices.append(i)
                else:
                    remaining_indices.append(i)
            
            sorted_indices = remaining_indices
        
        return selected_boxes, selected_scores, selected_labels

def apply_mask_pil_np(image_pil, mask_np, alpha=0.5, output_path=None):
    image_np = np.array(image_pil)
    if image_np.shape[-1] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    if mask_np.shape[1:]!=image_np.shape[:-1]:
        mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = np.zeros_like(image_np)
    overlay[mask_np > 0] = [0, 0, 255]
    blended = cv2.addWeighted(image_np, 1 - alpha, overlay, alpha, 0)
    
    return blended, mask_np

def apply_mask_np(image_np, mask_np, alpha=0.5, output_path=None):
    if image_np.shape[-1] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    if mask_np.shape[1:]!=image_np.shape[:-1]:
        mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = np.zeros_like(image_np)
    overlay[mask_np > 0] = [0, 0, 255]
    blended = cv2.addWeighted(image_np, 1 - alpha, overlay, alpha, 0)
    
    return blended, mask_np

def draw_bounding_box(image_array, boxes_np, color=(0, 0, 255), thickness=2):
    """
    Vẽ bounding box lên ảnh từ mảng NumPy và trả về hình ảnh.

    :param image_array: Mảng NumPy chứa ảnh (hình dạng: HxWxC).
    :param bbox: Bounding box dưới dạng (x1, y1, x2, y2).
    :param color: Màu sắc của bounding box (mặc định là đỏ).
    :param thickness: Độ dày của đường vẽ bounding box.
    :return: Hình ảnh đã vẽ bounding box.
    """
    # Sao chép ảnh gốc để không thay đổi ảnh ban đầu
    output_image = image_array.copy()

    # Vẽ tất cả bounding boxes
    for box in boxes_np:
        x1, y1, x2, y2 = map(int, box)  # Chuyển đổi tọa độ thành kiểu int
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)
        
        # Tính tâm của bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Vẽ tâm của bounding box
        cv2.circle(output_image, (center_x, center_y), radius=4, color=color, thickness=-1)  # -1 để vẽ hình tròn đặc

    return output_image  

def calculate_mask_bounding_box_area(mask):
    coords = np.column_stack(np.where(mask == 255))
    
    if coords.size == 0:
        return 0
    
    top, left = coords.min(axis=0)
    bottom, right = coords.max(axis=0)

    area = (right - left + 1) * (bottom - top + 1)
    return area

if __name__=="__main__":
    detect_mark = DetectedMarked(sam_path="checkpoints/sam2.1_hiera_base_plus.pt", sam_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml", 
                                 clipseg_path="weight/weights_40.pth", yolo_path="weight_yolo/best_crop_1.pt", device="cuda:0")

    image_path = "/TMTAI/AI_MemBer/workspace/tuannha/module_detect_marked/validation_marked/images/18_0_5_object.jpg"
    image = Image.open(image_path)
        
    result, mask = detect_mark.inference_clipseg(image=image)
    image_clipseg = draw_bounding_box(np.array(image), result)
    cv2.imwrite("test_clip.jpg", cv2.cvtColor(image_clipseg, cv2.COLOR_RGB2BGR))
    
    segment, mask_new = apply_mask_pil_np(image_pil=image, mask_np=mask)
    cv2.imwrite("segment.jpg", cv2.cvtColor(segment, cv2.COLOR_RGB2BGR))
    
    ##### Detection
    lst_bb = detect_mark.inference_detection(image=image)
    image_yolo = draw_bounding_box(np.array(image), lst_bb)
    cv2.imwrite("test_yolo.jpg", cv2.cvtColor(image_yolo, cv2.COLOR_RGB2BGR))
    
    # #### PIPELINE 1
    lst_bb, masks, points = detect_mark.inference_pipeline(image=image, is_multimask=True)
    image_clipseg = draw_bounding_box(np.array(image), lst_bb)
    cv2.imwrite("test_pipeline.jpg", cv2.cvtColor(image_clipseg, cv2.COLOR_RGB2BGR))
    
    image_point = np.array(image)
    for p in points:
        cv2.circle(image_point, (int(p[0]), int(p[1])), radius=4, color=(0, 0, 255), thickness=-1) 
        cv2.imwrite(f"test_pipeline_point.jpg", cv2.cvtColor(image_point, cv2.COLOR_RGB2BGR))
    if masks is not None:
        print(masks.shape)
        if len(masks.shape)==4:
            for id, mask in enumerate(masks):
                for i, m in enumerate(mask):
                    print(np.count_nonzero(m))
                    segment, mask_new = apply_mask_pil_np(image_pil=image, mask_np=m)
                    cv2.imwrite(f"segment_output/segment_pipeline_{id}_{i}.jpg", cv2.cvtColor(segment, cv2.COLOR_RGB2BGR))
        elif len(masks.shape)==3 and masks.shape[0]>1:
            for id, mask in enumerate(masks):
                segment, mask_new = apply_mask_pil_np(image_pil=image, mask_np=mask)
                cv2.imwrite(f"segment_output/segment_pipeline_{id}.jpg", cv2.cvtColor(segment, cv2.COLOR_RGB2BGR))
        else:
            segment, mask_new = apply_mask_pil_np(image_pil=image, mask_np=masks[0])
            cv2.imwrite(f"segment_output/segment_pipeline.jpg", cv2.cvtColor(mask_new*255, cv2.COLOR_RGB2BGR))
    


