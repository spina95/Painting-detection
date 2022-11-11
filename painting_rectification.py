import cv2
import numpy as np
from scipy.spatial import distance as dist

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def order_points_old(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def rectify(inputFrame, mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    approx = cv2.approxPolyDP(contours[0], 0.01 * cv2.arcLength(contours[0], True), True)
    #print("The mask polygon has " +str(len(approx)) +" sides")
    if len(approx) <= 6:
        approx = cv2.approxPolyDP(contours[0], 0.04 * cv2.arcLength(contours[0], True), True)
    m1 = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)

    # m2 = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    # c1 = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    # c2 = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)

    cv2.drawContours(m1, [approx], 0, 255, 3)
    # cv2.drawContours(m2, [approx], 0, 255, 3)

    box1 = cv2.minAreaRect(approx)
    box1 = cv2.boxPoints(box1)
    box1 = np.int0(box1)
    box1_o = order_points_old(box1)
    # print(box1)
    # print(box1_o)
    cv2.circle(m1, (box1_o[0][0], box1_o[0][1]), 3, 255, -1)
    cv2.circle(m1, (box1_o[1][0], box1_o[1][1]), 3, 255, -1)
    cv2.circle(m1, (box1_o[2][0], box1_o[2][1]), 3, 255, -1)
    cv2.circle(m1, (box1_o[3][0], box1_o[3][1]), 3, 255, -1)
    (tl, tr, br, bl) = box1_o
    # print((tl, tr, bl, br))
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # Check if the mid points are on the contour
    (x,y) =  (tltrX, tltrY)
    failedToFindPointOnContour = False
    while cv2.pointPolygonTest(approx, (x,y), False) < 0:
        y += 1
        if y >= mask.shape[0]:
            failedToFindPointOnContour = True
            break
    if not failedToFindPointOnContour:
        (tltrX, tltrY) = (x,y)

    (x, y) = (blbrX, blbrY)
    while cv2.pointPolygonTest(approx, (x, y), False) < 0:
        y -= 1
        if y <= mask.shape[0]:
            failedToFindPointOnContour = True
            break
    if not failedToFindPointOnContour:
        (blbrX, blbrY) = (x,y)

    (x, y) = (tlblX, tlblY)
    while cv2.pointPolygonTest(approx, (x, y), False) < 0:
        x += 1
        if x >= mask.shape[1]:
            failedToFindPointOnContour = True
            break
    if not failedToFindPointOnContour:
        (tlblX, tlblY) = (x,y)

    (x, y) = (trbrX, trbrY)
    while cv2.pointPolygonTest(approx, (x, y), False) < 0:
        x -= 1
        if x <= 0:
            failedToFindPointOnContour = True
            break
    if not failedToFindPointOnContour:
        (trbrX, trbrY) = (x,y)

    #Once

    # print((int(tltrX), int(tltrY)))
    # print((int(blbrX), int(blbrY)))
    # print((int(tlblX), int(tlblY)))
    # print((int(trbrX), int(trbrY)))

    cv2.circle(m1, (int(tltrX), int(tltrY)), 8, 255, -1)
    cv2.circle(m1, (int(blbrX), int(blbrY)), 8, 255, -1)
    cv2.circle(m1, (int(tlblX), int(tlblY)), 8, 255, -1)
    cv2.circle(m1, (int(trbrX), int(trbrY)), 8, 255, -1)

    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # cv2.circle(m1, (box1_o[0][0], box1_o[0][1]), 8, 255, -1)
    cv2.drawContours(m1, [box1], 0, 255, 1)

    # box2 = cv2.minAreaRect(approx)
    # box2 = cv2.boxPoints(box2)
    # box2 = np.int0(box2)
    # box2_o = order_points_old(box2)
    # cv2.drawContours(m2, [box2], 0, 255, 1)

    # Compute center of shapes
    # M1 = cv2.moments(contours[0])
    # cX1 = int(M1["m10"] / M1["m00"])
    # cY1 = int(M1["m01"] / M1["m00"])
    #
    # M2 = cv2.moments(approx)
    # cX2 = int(M2["m10"] / M2["m00"])
    # cY2 = int(M2["m01"] / M2["m00"])

    # cv2.circle(m1, (int(tltrX), int(tltrY)), 3, 255, -1)
    # cv2.circle(m1, (int(blbrX), int(blbrY)), 3, 255, -1)
    # cv2.circle(m1, (int(tlblX), int(tlblY)), 3, 255, -1)
    # cv2.circle(m1, (int(trbrX), int(trbrY)), 3, 255, -1)

    # y = int(cYO - int(dA/2) -50)
    h = int(dA)
    # x = int(cXO - int(dB/2) -50)
    w = int(dB)
    # (tl, tr, br, bl) = box1_o

    if h > 600 or w > 600:
        h = int(h * 0.4)
        w = int(w * 0.4)

    p11 = None
    p21 = None
    is_rect = False
    if len(approx) == 4:
        approx_l = np.squeeze(approx, 1)
        sorted_approx = order_points_old(approx_l)
        (tl1, tr1, br1, bl1) = sorted_approx

        h = int(dist.euclidean((int(tl1[0]), int(tl1[1])), (int(bl1[0]), int(bl1[1]))))
        w = int(dist.euclidean((int(tl1[0]), int(tl1[1])), (int(tr1[0]), int(tr1[1]))))

        if h > 600 or w > 600:
            h = int(h*0.4)
            w = int(w*0.4)

        cv2.circle(m1, (int(tl1[0]), int(tl1[1])), 10, 255, -1)
        cv2.circle(m1, (int(tr1[0]), int(tr1[1])), 10, 255, -1)
        cv2.circle(m1, (int(br1[0]), int(br1[1])), 10, 255, -1)
        cv2.circle(m1, (int(bl1[0]), int(bl1[1])), 10, 255, -1)
        p11 = np.float32([ [int(tl1[0]), int(tl1[1])], [int(tr1[0]), int(tr1[1])], [int(br1[0]), int(br1[1])], [int(bl1[0]), int(bl1[1])]])
        p21 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        is_rect = True

    else:
        p11 = np.float32(
            [[int(tltrX), int(tltrY)], [int(blbrX), int(blbrY)], [int(tlblX), int(tlblY)], [int(trbrX), int(trbrY)]])
        p21 = np.float32([[int(w / 2), 0], [int(w / 2), h], [0, int(h / 2)], [w, int(h / 2)]])
        # p11 = np.float32([[tl[0], tl[1]], [tr[0], tr[1]], [int(tlblX), int(tlblY)], [int(trbrX), int(trbrY)]])
        # p21 = np.float32([[int(w / 2), 0], [int(w / 2), h], [0, int(h / 2)], [w, int(h / 2)]])
        #
    matrix1 = cv2.getPerspectiveTransform(p11, p21)
    imc = inputFrame.copy()
    imc[mask == 0] = (0, 0, 0)
    r1 = cv2.warpPerspective(imc, matrix1, (w, h))

    # cv2.circle(m1, (cX1, cY1), 1, 255, -1)
    # cv2.circle(m2, (cX2, cY2), 1, 255, -1)

    # print((cX1, cY1))
    # cv2.imshow('m2', m2)
    #m1_r = cv2.resize(m1, (int(inputFrame.shape[1] * 0.4), int(inputFrame.shape[0] * 0.4)), interpolation=cv2.INTER_AREA)
    #cv2.imshow('m1', m1_r)
    # cv2.imshow('r1', r1)
    # cv2.imshow('c1', c1)
    # cv2.imshow('c2', c2)
    #key = cv2.waitKey(0)
    return r1, p11, is_rect