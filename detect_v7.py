import time
import cv2
import torch
from Detect_yolov7 import Detect
from utils.general import scale_coords
import numpy as np
import matplotlib.path as mplPath
from notification_phone import pushbullet_noti
from datetime import datetime
import os

ROOT= os.getcwd()

def calculate_distance(point1, point2):
    # '''Calculate usual distance.'''
    x1, y1 = point1
    x2, y2 = point2
    return np.linalg.norm([x1 - x2, y1 - y2])

def convert_to_bird(centers, M):
    # '''Apply the perpective to the bird's-eye view.'''
    centers = [cv2.perspectiveTransform(np.float32([[center]]), M) for center in centers.copy()]
    centers = [list(center[0, 0]) for center in centers.copy()]
    return centers


def bird_detect_people_on_frame(img, xyxy ,dangerus, confidence, distance, width, height,
                                region=None, dst=None):
    # xyxy = results.xyxy[0].cpu().numpy()  # xyxy are the box coordinates
    #          x1 (pixels)  y1 (pixels)  x2 (pixels)  y2 (pixels)   confidence        class
    # tensor([[7.47613e+02, 4.01168e+01, 1.14978e+03, 7.12016e+02, 8.71210e-01, 0.00000e+00],
    #         [1.17464e+02, 1.96875e+02, 1.00145e+03, 7.11802e+02, 8.08795e-01, 0.00000e+00],
    #         [4.23969e+02, 4.30401e+02, 5.16833e+02, 7.20000e+02, 7.77376e-01, 2.70000e+01],
    #         [9.81310e+02, 3.10712e+02, 1.03111e+03, 4.19273e+02, 2.86850e-01, 2.70000e+01]])
    # print(xyxy)
    xyxy = xyxy[xyxy[:, 4] >= confidence]  # Filter desired confidence  xác suất các class
    # lấy tọa độ class 4
    xyxy = xyxy[xyxy[:, 5] == 4]  # Consider only people
    xyxy = xyxy[:, :4]    # lấy tọa độ 4 thông số đầu


    # Calculate the centers of the circles
    # They will be the centers of the bottom of the boxes
    centers = []
    for x1, y1, x2, y2 in xyxy:
        center = [np.mean([x1, x2]), y2]
        centers.append(center)
    # print(centers[0][0])
    # print(int(centers[0][1]))
    # tensor([[ 70.00000, 154.00000, 590.00000, 259.00000,   0.94268,   0.00000],
    #     [549.00000, 155.00000, 633.00000, 424.00000,   0.94008,   4.00000]])
    #     591.0
    #     424

    # We create two transformations
    if region is None:
        # The region on the original image
        region = np.float32([[144, 130], [222, 129], [width, height], [0, height]])

    if dst is None:
        # The rectangle we want the image to be trasnformed to
        dst = np.float32([[0, 0], [width, 0], [width, 3*width], [0, 3*width]])

    # The first transformation is straightforward: the region to the rectangle
    # as thin the example before
    M = cv2.getPerspectiveTransform(region, dst)

    # The second transformation is a trick, because, using the common transformation,
    # we can't draw circles at left of the region.
    # This way, we flip all things and draw the circle at right of the region,
    # because we can do it.
    region_flip = region*np.float32([-1, 1]) + np.float32([width, 0])
    dst_flip = dst*np.float32([-1, 1]) + np.float32([width, 0])
    M_flip = cv2.getPerspectiveTransform(region_flip, dst_flip)

    # Convert to bird
    # Now, the center of the circles will be positioned on the rectangle
    # and we can calculate the usual distance
    bird_centers = convert_to_bird(centers, M)
    
    # We verify if the circles colide
    # If so, they will be red
    colors = ['green']*len(bird_centers)
    if(dangerus):
        colors=['red']*len(bird_centers)
    
    for i in range(len(bird_centers)):
        for j in range(i+1, len(bird_centers)):
            dist = calculate_distance(bird_centers[i], bird_centers[j])
            if dist < distance:
                colors[i] = 'red'
                colors[j] = 'red'

    # We draw the circles
    # Because we have two transformation, we will start with two empty
    # images ("overlay" images) to draw the circles
    overlay = np.zeros((3*width, 4*width, 3), np.uint8)
    overlay_flip = np.zeros((3*width, 4*width, 3), np.uint8)
    for i, bird_center in enumerate(bird_centers):
        if colors[i] == 'green':
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        x, y = bird_center
        x = int(x)
        y = int(y)

        if x >= int(distance/2+20/2):
            # If it's the case the circle is inside or at right of our region
            # we can use the normal overlay image
            overlay = cv2.circle(overlay, (x, y), int(distance/2),
                                  color, 15, lineType=cv2.LINE_AA)
        else:
            # If the circle is at left of the region,
            # we draw the circle inverted on the other overlay image
            x = width - x
            overlay_flip = cv2.circle(overlay_flip, (x, y), int(distance/2),
                                  color, 15, lineType=cv2.LINE_AA)

    # We apply the inverse transformation to the overlay
    overlay = cv2.warpPerspective(overlay, M, (width, height),
                                  cv2.INTER_NEAREST, cv2.WARP_INVERSE_MAP)
    # We apply the inverse of the other transformation to the other overlay
    overlay_flip = cv2.warpPerspective(overlay_flip, M_flip, (width, height),
                                       cv2.INTER_NEAREST, cv2.WARP_INVERSE_MAP)
    # Now we "unflip" what the second overlay
    overlay_flip = cv2.flip(overlay_flip, 1)

    # We add all images
    img = cv2.addWeighted(img, 1, overlay, 1, 0)
    img = cv2.addWeighted(img, 1, overlay_flip, 1, 0)
    return img

mouse = []
def draw_(event,x,y,flags,param):         # bát sự kiện chuột lấy tọa độ
    global mouse
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouse= mouse + [[x,y]]

center_baby = (0,0)

if __name__ == '__main__':
    print("Chọn mode :")
    print("1: Indoor")
    print("2: outdoor")
    mode_detect = int(input())
    if(mode_detect  == 1) :
        weights = ROOT + r"\best_model\v7_tiny.pt"
        im_size = 640
        conf_thres = 0.6
        iou_thres = 0.6
        device = 'cpu'
        classes = None
        vid = cv2.VideoCapture(ROOT + r"\video\video4.mp4")

        fps = vid.get(cv2.CAP_PROP_FPS)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        confidence=0.9
        distance=60

        size = (width, height)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # if os.path.exists('output.avi'):
        #     os.remove('bird_output.avi')
        
        result = cv2.VideoWriter('runs\detect\output.mp4',fourcc, fps, (width,height))
        
        # load model
        Det = Detect(weights ,im_size, device)
        points = []
        # center_list = []
        detect = False
        dis = False

        st1 = time.time()
        is_detc = False
        dangerus = 0
        while True:
            with torch.no_grad(): 
                _,frame = vid.read()   
                if not _:
                    break
                if is_detc:
                    st = time.time()
                    # center_list = []
                    pred = Det.detect(frame ,conf_thres = conf_thres, iou_thres=iou_thres) # result 

                    # check pred is None or not None?
                    if pred != torch.tensor([]):
                        if list(pred[0]) != []: 
                            # pr = time.time()
                            # rescale boundingbox
                            pred_rescale = torch.tensor(pred[0])
                            pred_rescale = scale_coords(Det.img_size_detect[2:], pred_rescale, frame.shape).round()

                            # get center bounding box
                            centers = Det.get_center(pred_rescale)[:,:2]
                            center_draw = tuple(centers.numpy()[0])
                            centers = list(centers.numpy()[0])

                            # get center bottom bounding box
                            centerbottom = Det.get_center_bottom(pred_rescale)[:,:2]
                            centerbot_draw = tuple(centerbottom.numpy()[0])
                            centers_bot = list(centerbottom.numpy()[0])    

                            # initial 2 lists of people and objects
                            list_peo = []
                            list_obj = []

                            for j in range(len(pred[0])):
                                if((int(pred[0][j][5])) ==4):
                                    coordinatesX = int(centerbottom[j][0])
                                    coordinatesY = int(centerbottom[j][1])
                                    center_peo = [coordinatesX,coordinatesY]
                                    list_peo.append(center_peo)
                                    cv2.circle(frame,(int(coordinatesX),int(coordinatesY)),5, (255,0,0), 5)

                                    # if inside dangerous zone send notification to telegram  
                                    # if Det.sInside(points,center_peo):
                                    #     frame = Det.alert(frame)
                                        
                                elif((int(pred[0][j][5])) ==6):
                                    pass

                                else:
                                    coordX = int(centerbottom[j][0])
                                    coordY = int(centerbottom[j][1])
                                    center_obj = [coordX,coordY]
                                    list_obj.append(center_obj)
                                    cv2.circle(frame,(int(coordX),int(coordY)),4, (0,0,255), 5)
                            # cv2.circle(frame,centerbot_draw,radius=5,color=(255,0,0),thickness=5)

                            # Convert original coordinates to bird coordinates
                            frame = Det.bird_detect_people(frame,list_peo,list_obj,350,width,height) 

                            
                    print('time per frame: ',time.time()-st)
                    print('FPS: ',fps)
                    # print(time.time()-pr)

                    # draw bounding box
                    # img_rstl = Det.draw_all_box(img=frame,pred=pred)
                    # img_rstl = frame   
                else:
                    img_rstl = height

                event =  cv2.waitKey(1)

                if event== ord('q'):
                    break
                elif event == ord('p'):
                    cv2.waitKey(-1) 
                elif event == ord('c'):
                    print("c")
                    is_detc = False
                elif event == ord('d'):
                    # points.append(points[0])
                    # points = points[:-1]
                    print("d")
                    is_detc = True 

                # write frame to video output file
                result.write(frame)

                # show window
                cv2.imshow("Warning", frame)
        # When everything done, release the capture
        vid.release()
        result.release()
        cv2.destroyAllWindows()
        
    if (mode_detect == 2):
        weights = ROOT + r"\best_model\fake_bacony.pt"
        im_size = 640
        conf_thres = 0.6
        iou_thres = 0.6
        device = 'cpu'
        classes = None
        vid = cv2.VideoCapture(ROOT + r"\video\video2.mp4")

        fps = vid.get(cv2.CAP_PROP_FPS)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        confidence=0.9
        distance=60

        size = (width, height)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # if os.path.exists('output.avi'):
        #     os.remove('bird_output.avi')
        
        result = cv2.VideoWriter('runs\detect\output.mp4',fourcc, fps, (width,height))
        
        # load model
        Det = Detect(weights ,im_size, device)
        points = []
        # center_list = []
        detect = False
        dis = False

        st1 = time.time()
        is_detc = True
        dangerus = 0
        while True:
            with torch.no_grad(): 
                _,img = vid.read()   
                if not _:
                    break
                if is_detc:
                    st = time.time()
                    pred = Det.detect(img ,conf_thres = conf_thres, iou_thres=iou_thres) # result 
                    # print(pred)
                    # [tensor([[ 73.26988, 122.54459, 460.91907, 202.68332,   0.96287,   0.00000],
                    #          [437.52539, 133.90805, 509.43604, 341.43826,   0.96000,   4.00000]])]
                    pr = time.time()
                    # rescale bounding boxes
                    pred_rescale = torch.tensor(pred[0])
                    xyxy = pred_rescale
                    # print(pred_rescale)
                    # tensor([[ 55.13585, 128.40811, 443.07941, 205.89993,   0.96077,   0.00000],
                    #         [381.29095, 116.96262, 440.59949, 304.08832,   0.92224,   4.00000]])

                    pred_rescale[:, :4] = scale_coords(Det.img_size_detect[2:], pred_rescale[:, :4], img.shape).round()
                    
                    # get center point
                    center = Det.get_center(pred_rescale[:10])[:,:2]  #pred_rescale[:4]
                    # draw centrer point obj baby
                    for j in range(len(pred[0])):
                        if((int(pred[0][j][5])) == 4):
                            coordinatesX = int(center[j][0])
                            coordinatesY = int(center[j][1])
                            center_baby = (coordinatesX,coordinatesY)
                            cv2.circle(img,(coordinatesX,coordinatesY),6, (55,160,15), 5)
                    
                    # get class
                    _type = pred_rescale[:,5]
                    
                    print("Time detect 1 frame: " + str(time.time()-st))
                    img_rstl = Det.draw_all_box(img=img,pred=pred) # draw box
                    # img_rstl = img
                    img_rstl = bird_detect_people_on_frame(img_rstl, xyxy,dangerus,confidence, distance,width , height)
                    dangerus = 0
                    # img_rstl = img
                else:
                    img_rstl = img

            # draw polylines dangerus
                # print(mouse)
                # print(len(mouse))

                if len(mouse) >=3 :
                    pts = np.array(mouse, np.int32)
                    pts = pts.reshape((-1,1,2))
                    cv2.polylines (img_rstl, [pts], True, (0,255,255),5)

                    mouse_path = mplPath.Path(mouse)
                    # # kiểm tra nguy hiểm ( center có nằm trong vùng nguy hiểm k)
                    if mouse_path.contains_point(center_baby):
                        dangerus = 1 
                        print("\t\t Dangerus ")
                        # print("Time st1 :" + str(st1) +"       " + str(time.time()))
                        font = cv2.FORMATTER_FMT_DEFAULT
                        cv2.putText(img_rstl,'Dangerus',(10 , 80), font, 3,(0,0,255),10,cv2.LSD_REFINE_ADV)
                        cv2.circle(img_rstl,(coordinatesX,coordinatesY),6, (0,0,255), 5)
                        if time.time() - st1 > 20 :
                            # lấy ngày giờ hiện tại
                            now = datetime.now() 
                            # dd/mm/YY H:M:S
                            dt_string = str(now.strftime("%d/%m/%Y %H:%M:%S"))
                            pushbullet_noti("Dangerous","Baby is near dangerous objects : " + dt_string)
                            st1=time.time()
                            print("Sent notify phone seccessful")

                cv2.imshow("Camera", img_rstl)
                cv2.setMouseCallback("Camera",draw_)
                event =  cv2.waitKey(1)
                if  event== ord('q'):
                    break
                elif event == ord(' '):
                    is_detc = not is_detc
    else:
        print("Error")
                