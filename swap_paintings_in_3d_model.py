import cv2
import os
from painting_retrieval import PaintingFinder
from detectron2_paintings_detection import detect_paintings
import numpy as np
from painting_rectification import rectify

def unrectify(inputFrame, mask, similarImage, originalImagePointsForProspectiveTransformation, is_rect):
    sH = similarImage.shape[0]
    sW = similarImage.shape[1]
    ifb = inputFrame.copy()
    ifb[mask == 255] = 0
    # similarImage = cv2.cvtColor(similarImage,cv2.COLOR_BGR2RGB)
    # inputFrame = cv2.cvtColor(inputFrame,cv2.COLOR_BGR2RGB)

    # rows, cols, ch = inputFrame.shape
    p11 = np.float32([[int(sW / 2), 0], [int(sW / 2), sH], [0, int(sH / 2)], [sW, int(sH / 2)]])
    p21 = originalImagePointsForProspectiveTransformation

    if is_rect:
        p11 = np.float32([[0, 0], [sW, 0], [sW, sH], [0, sH]])
        p21 = originalImagePointsForProspectiveTransformation

    matrix = cv2.getPerspectiveTransform(p11, p21)
    r1 = cv2.warpPerspective(similarImage, matrix, (inputFrame.shape[1], inputFrame.shape[0]))
    overlay = cv2.add(ifb, r1)

    # width = int(image.shape[1] * 35 / 100)
    # height = int(image.shape[0] * 35 / 100)
    # dim = (width, height)
    # r1_r = cv2.resize(overlay, dim, interpolation=cv2.INTER_AREA)
    # cv2.imshow('r1', r1_r)

    return overlay

def loadPaintingsDB(DBPath):
    paintingsDB = {}
    for imagefile in os.listdir(DBPath):
        tmpImg = cv2.imread(os.path.join(DBPath, imagefile))
        tmpImgRGB = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2RGB)
        tmpImgGray = cv2.cvtColor(tmpImgRGB, cv2.COLOR_RGB2GRAY)
        paintingsDB[imagefile] = (tmpImg, tmpImgRGB, tmpImgGray)
    return paintingsDB

def swap_paintings_3d_model():
    pf = PaintingFinder()
    paintingsDB = loadPaintingsDB("paintings_db")

    d_model_schreeshots_images_path = "screenshots_3d_model"

    for imagefile in os.listdir(d_model_schreeshots_images_path):
        image = cv2.imread(os.path.join(d_model_schreeshots_images_path, imagefile))
        scale_percent = 40  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        # resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        outim, triple_paintings_boxes_masks = detect_paintings(image, True)
        unrectified = None
        c = 0
        for tpbm in triple_paintings_boxes_masks:
            mask = tpbm[2]
            #if c <= 3:
                #c += 1
               # continue

            # cv2.imshow('Mask', mask)
            # key = cv2.waitKey(0)
            rectified, originalImagePointsForProspectiveTransformations, is_rect = rectify(image, mask)

            # cv2.imshow('Rectified', rectified)
            # key = cv2.waitKey(0)

            similars = pf.getRankedListOfSimilars([rectified], paintingsDB, 2)
            if similars is not None and len(similars) > 0:
                mostSimilar = list(similars[0].keys())[0]
                msImage = paintingsDB[mostSimilar][0]
                if unrectified is None:
                    unrectified = unrectify(image, mask, msImage, originalImagePointsForProspectiveTransformations, is_rect)
                else:
                    unrectified = unrectify(unrectified, mask, msImage, originalImagePointsForProspectiveTransformations, is_rect)
                resized_ur = cv2.resize(unrectified, dim, interpolation=cv2.INTER_AREA)
                #cv2.imshow('Unrectified Frame', resized_ur)
                #key = cv2.waitKey(0)
                # cv2.imshow('Most Similar', msImage)
                # key = cv2.waitKey(0)

            #print('')

        resized_org = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        resized_unrectified = cv2.resize(unrectified, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow('Frame with original paintings', resized_org)
        cv2.imshow('Frame with swapped paintings', resized_unrectified)

        key = cv2.waitKey(0)
