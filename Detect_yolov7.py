from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device
import cv2
import torch
# from models.common import DetectMultiBackend
import datetime
# from send_telegram import send_warning
import threading
import itertools
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point

class Detect:
    def __init__(self, weights, img_size = 640, device ='cpu' ):
        # Initialize
        set_logging()
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        self.last_alert = None
        self.region = None
        self.dst = None
        self.alert_telegram_each = 30
        self.list = []
        
        # Load model
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(img_size, s=self.stride)  # check img_size


        if self.half:
            self.model.half()  # to FP16
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
    
    # Using Euclid algorithm
    def distance(self,point1, point2):
        '''Calculate usual distance.'''
        x1, y1 = point1
        x2, y2 = point2
        return np.linalg.norm([x1 - x2, y1 - y2])
    
    # Convert original coordinates to bird-view coordinates
    def convert_to_bird(self,centers, M):
        '''Apply the perpective to the bird's-eye view.'''
        centers = [cv2.perspectiveTransform(np.float32([[center]]), M) for center in centers.copy()]
        centers = [list(center[0, 0]) for center in centers.copy()]
        return centers
    
    # Take 4 points and output the corresponding matrix
    def PerspectiveTransform(self,region, dst):
        M = cv2.getPerspectiveTransform(region, dst)
        return M
    

    # When using the common transformation, it can't draw circles at left of the region.
    # Therefore, we flip frame and draw the circle at right of the region
    def FlipFrame(self,region,dst,width):
        region_flip = region*np.float32([-1, 1]) + np.float32([width, 0])
        dst_flip = dst*np.float32([-1, 1]) + np.float32([width, 0])
        M_flip = cv2.getPerspectiveTransform(region_flip, dst_flip)

        return M_flip
    
    # Detect dangerous objects zone and send warning
    def detect_dangerous(self,frame,list1,list2,distance,colors1,colors2):
        for i in range(len(list1)):
            for j in range(len(list2)):
                dist = self.distance(list1[i], list2[j])
                if dist < distance:
                    colors1[i] = 'red'
                    colors2[j] = 'red'
                    cv2.putText(frame, "Dangerous!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # self.alert(frame)
        return frame
    
# def pushllet():


    def color_object(self,frame,list,colors,distance,width,overlay,overlay_flip):
        for i,bird_center in enumerate(list):
            if colors[i] == 'green':
                color = (0, 255, 0)
            elif colors[i] == 'red':
                color = (0, 0, 225)
            else:
                color = (255, 255, 0)
            x, y = bird_center
            x = int(x)
            y = int(y)
            if x >= int(distance/2+15/2):
                overlay = cv2.circle(overlay, (x, y), int(distance/2),
                                    color, 15, lineType=cv2.LINE_AA)

            else:
                x = width - x
                overlay_flip = cv2.circle(overlay_flip, (x, y), int(distance/2),
                                    color, 15, lineType=cv2.LINE_AA)
                
        return frame

    # Detect and handles on bird view
    def bird_detect_people(self,frame,list1, list2, distance, width, height,
                                region=None, dst=None):
        
        print('list_peo: ',list1)
        print('list_obj: ',list2)
        if self.region is None:
            # The rectangle on the original frame

            region = np.float32([[300, 210], [520, 210], [width, height], [0, height]])

            # Draw the region on frame
            region1 = np.array(region, dtype=np.int32).reshape((-1, 1, 2))
            frame = cv2.polylines(frame.copy(), [region1], True, (0, 0, 255), 2)
        if self.dst is None:
            # The rectangle be trasnformed 
            dst = np.float32([[0, 0], [width, 0], [width, 3*width], [0, 3*width]])
            # print(dst)

        # Call convert matrix and flip convert matrix
        M= self.PerspectiveTransform(region,dst)
        M_flip = self.FlipFrame(region,dst,width)

        # Convert coordinates to bird coordinates and draw bounding circles
        bird_centers_peo = self.convert_to_bird(list1 , M)
        bird_centers_obj = self.convert_to_bird(list2 , M)

        # Setup the colors for each circle
        colors1 = ['green']*len(bird_centers_peo) # green for people
        colors2 = ['blue']*len(bird_centers_obj) # blue for objects

        # Detect dangerous objects zone and warning 
        frame = self.detect_dangerous(frame,bird_centers_peo,bird_centers_obj,distance,colors1,colors2)
        print('color1: ',colors1)
        print('color2: ',colors2)
        overlay = np.zeros((3*width, 4*width, 3), np.uint8)
        overlay_flip = np.zeros((3*width, 4*width, 3), np.uint8)

        
        # Chane colors bounding circle 
        frame = self.color_object(frame,bird_centers_peo,colors1,distance,width,overlay,overlay_flip)
        frame = self.color_object(frame,bird_centers_obj,colors2,distance,width,overlay,overlay_flip)
        
        
        # Apply the inverse transformation to the overlay
        overlay = cv2.warpPerspective(overlay, M, (width, height),
                                    cv2.INTER_NEAREST, cv2.WARP_INVERSE_MAP)
        
        # Apply the inverse of the other transformation to the other overlay
        overlay_flip = cv2.warpPerspective(overlay_flip, M_flip, (width, height),
                                        cv2.INTER_NEAREST, cv2.WARP_INVERSE_MAP)
        # Unflip what the second overlay
        overlay_flip = cv2.flip(overlay_flip, 1)
      
        # Add all images
        frame = cv2.addWeighted(frame, 1, overlay, 1, 0)
        frame = cv2.addWeighted(frame, 1, overlay_flip, 1, 0)

        return frame

    # Choose points thround left click mouse
    def handle_left_click(self,event, x, y,flags, points):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])

    # Draw polygon through previously selected points
    def draw_polygon (self,frame, points):
        for point in points:
            frame = cv2.circle( frame, (point[0], point[1]), 5, (0,0,255), -1)

        frame = cv2.polylines(frame, [np.int32(points)], False, (255,0, 0), thickness=2)
        return frame

    # Check if the point is in the region or not?
    def isInside(self,points, centroid):
        polygon = Polygon(points)
        centroid = Point(centroid)
        return polygon.contains(centroid)

    # Send notification through telegram
    # def alert(self, img):
    #     cv2.putText(img, "Dangerous!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #     # New thread to send telegram after 30 seconds
    #     if (self.last_alert is None) or (
    #             (datetime.datetime.utcnow() - self.last_alert).total_seconds() > self.alert_telegram_each):
    #         self.last_alert = datetime.datetime.utcnow()
    #         cv2.imwrite("alert.png", cv2.resize(img, dsize=(920,640), fx=0.2, fy=0.2))
    #         thread = threading.Thread(target=send_warning("alert.png"))
    #         thread.start()
    #     return img
    
    def precess_img(self, img, size):
        '''Resize image and convert to tensor 4D'''
        img = letterbox(img,size)[0]
        img_norm = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = img_norm.astype(np.float32)/255
        img_norm = np.transpose(img_norm, (2, 0, 1))   
        return torch.from_numpy(img_norm).unsqueeze(0)

        
    def detect(self, image, conf_thres = 0.25, iou_thres=0.45, classes=None, agnostic_nms = False, augment =False):
        img = self.precess_img(image,(640,640))
        self.img_size_detect = img.shape
        pred = self.model(img, augment=augment)[0]  #detect
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        return pred

    # Get center bounding box
    def get_center(self, box):
        y = box.clone() if isinstance(box, torch.Tensor) else np.copy(box)
        y[:, 0] = (box[:, 0] + box[:, 2]) / 2  # x center
        # y[:, 1] = (box[:, 1] + box[:, 3]) / 2  # y center
        y[:, 1] = (box[:, 3]) # y center
        return y
    
    # Get center bottom of bounding box
    def get_center_bottom(self, box):
        y = box.clone() if isinstance(box, torch.Tensor) else np.copy(box)
        y[:, 0] =  (box[:, 0] + box[:, 2]) / 2  # x center
        y[:, 1] =  (box[:, 3])  # y bottom 
        return y
    
    def get_all_center(self, all_box):
        center = torch.tensor([])
        for i in all_box:
            center = np.append(center, self.get_center(i))
        

    def draw_all_box(self,img, pred):
        # Process detections
        for _, det in enumerate(pred):  # detections per image
            s, im0, ='', img
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(self.img_size_detect[2:], det[:, :4], im0.shape).round()
                # print(self.img_size_detect[2:])
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
        return im0
    