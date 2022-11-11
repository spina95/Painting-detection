import argparse
import os
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from numpy import random
from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

def plot_one_box(x, img, color=None, label=None, line_thickness=1):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, tl, cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def detect(input_frame, returnPicturesWithROI = False, image_title='out.png', save_img=True, device='', lWeights='yolov5_weights/yolov5s_persons_busts_paintings_best.pt', output_folder='inference/output', source_folder='inference/images', show_img=True, img_size=640, augment=False, agnostic_nms=False, classes=None, iou_thres=0.5, conf_thres=0.5):
    out = output_folder
    source = source_folder
    weights = lWeights
    view_img = show_img
    imgsz = img_size
    drawings_in_frame = []
    drawings_rois_in_frame = []
    persons_in_frame = []
    persons_rois_in_frame = []
    busts_in_frame = []
    busts_rois_in_frame = []

    input_frame_0 = input_frame.copy()

    # Padded resize
    input_img = letterbox(input_frame, new_shape=img_size)[0]

    # Convert
    input_img = input_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    input_img = np.ascontiguousarray(input_img)

    # Initialize
    device = select_device(device)
    #if os.path.exists(out):
     #   shutil.rmtree(out)  # delete output folder
    #os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    #_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    # for path, img, im0s, vid_cap in dataset:
    path = output_folder
    img = torch.from_numpy(input_img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t2 = time_synchronized()

    # Process detections
    out_frame = input_frame
    for i, det in enumerate(pred):  # detections per image
        if det == None:
            continue
        p, s, im0 = path, '', input_frame
        save_path = str(Path(out) / image_title)
        # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # Print results
        for c in det[:, -1].detach().unique():
            n = (det[:, -1] == c).sum()  # detections per class
        s += '%g %ss, ' % (n, names[int(c)])  # add to string

        # Write results
        for *xyxy, conf, cls in det:
            if save_img or view_img:
                # Add bbox to image
                label = '%s %.2f' % (names[int(cls)], conf)


                y1 = int(xyxy[1] - 10)
                y2 = int(xyxy[3] + 10)
                x1 = int(xyxy[0] - 10)
                x2 = int(xyxy[2] + 10)

                if y1 < 0:
                    y1 = 0
                if y2 > input_frame_0.shape[0]:
                    y2 = input_frame_0.shape[0]
                if x1 < 0:
                    x1 = 0
                if x2 > input_frame_0.shape[1]:
                    x2 = input_frame_0.shape[1]

                res = input_frame_0[y1:y2, x1:x2].copy()


                if int(cls) == 0:
                    drawings_in_frame.append(res)
                    drawings_rois_in_frame.append((x1,y1,x2,y2))
                    #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                elif int(cls) == 1:
                    persons_in_frame.append(res)
                    persons_rois_in_frame.append((x1,y1,x2,y2))
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                elif int(cls) == 2:
                    busts_rois_in_frame.append((x1,y1,x2,y2))
                    busts_in_frame.append(res)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
        out_frame = im0
        # Print time (inference + NMS)
        print('%sDone. (%.3fs)' % (s, t2 - t1))

        # Stream results
        if view_img:
            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

        # Save results (image with detections)
        #if save_img:
            #cv2.imwrite(save_path, im0)
    print('Done. (%.3fs)' % (time.time() - t0))
    return persons_in_frame, busts_in_frame, persons_rois_in_frame, busts_rois_in_frame, out_frame
