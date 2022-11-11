import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import tensorflow.compat.v1 as tf
from gaze_utils.head_pose_estimation import CnnHeadPoseEstimator
import random

# Function used to get the rotation matrix
def yaw2rotmat(yaw):
    x = 0.0
    y = 0.0
    z = yaw
    ch = np.cos(z)
    sh = np.sin(z)
    ca = np.cos(y)
    sa = np.sin(y)
    cb = np.cos(x)
    sb = np.sin(x)
    rot = np.zeros((3, 3), 'float32')
    rot[0][0] = ch * ca
    rot[0][1] = sh * sb - ch * sa * cb
    rot[0][2] = ch * sa * sb + sh * cb
    rot[1][0] = sa
    rot[1][1] = ca * cb
    rot[1][2] = -ca * sb
    rot[2][0] = -sh * ca
    rot[2][1] = sh * sa * cb + ch * sb
    rot[2][2] = -sh * sa * sb + ch * cb
    return rot

def getRandomRGBColor(seed):
    r = random.randint(seed, 255)
    g = random.randint(seed, 255)
    b = random.randint(seed, 255)
    rgb = [r, g, b]
    return rgb

def detect_people_watching_paintings_with_head_poze_estimetion(frame_with_rois, frame, people_roi, paintings_roi, retrieval, printTitlesOfPaintingsWatched=True):
    colorsAlreadyUsed = []
    cp = frame_with_rois.copy()
    for index, roi in enumerate(people_roi):
        x1, y1, x2, y2 = roi
        h = y2 - y1
        w = x2 - x1

        xHead = int((x1 + int(w / 2)))
        yHead = int((y1 + 50))

        person_gaze_lines_color = getRandomRGBColor(index)
        while person_gaze_lines_color in colorsAlreadyUsed:
            person_gaze_lines_color = getRandomRGBColor(index)
        colorsAlreadyUsed.append(person_gaze_lines_color)

        print('width: ' + str(w))
        
        image = frame[y1:y1 + h, x1:x1 + w]

        # Load the cascade
        face_cascade = cv2.CascadeClassifier('gaze_utils/etc/xml/haarcascade_frontalface_default.xml')

        # Convert into grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        direction = ""
        if len(faces) != 0:
            for (x, y, w, h) in faces:
                face = image[y:y+h, x:x+w]
                if face.shape[0] > 64 and face.shape[1] > 64 and face.shape[0] == face.shape[1]:
                    print(image.shape)
                    tf.compat.v1.disable_eager_execution()
                    sess = tf.Session()  # Launch the graph in a session.
                    my_head_pose_estimator = CnnHeadPoseEstimator(sess)  # Head pose estimation object

                    # Load the weights from the configuration folders
                    my_head_pose_estimator.load_roll_variables(
                        os.path.realpath("gaze_utils/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"))
                    my_head_pose_estimator.load_pitch_variables(
                        os.path.realpath("gaze_utils/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))
                    my_head_pose_estimator.load_yaw_variables(
                        os.path.realpath("gaze_utils/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf"))

                    cam_w = face.shape[1]
                    cam_h = face.shape[0]
                    c_x = cam_w / 2
                    c_y = cam_h / 2
                    f_x = c_x / np.tan(60 / 2 * np.pi / 180)
                    f_y = f_x
                    camera_matrix = np.float32([[f_x, 0.0, c_x],
                                                [0.0, f_y, c_y],
                                                [0.0, 0.0, 1.0]])
                    print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")
                    # Distortion coefficients
                    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
                    # Defining the axes
                    axis = np.float32([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.5]])

                    roll_degree = my_head_pose_estimator.return_roll(face,
                                                                    radians=False)  # Evaluate the roll angle using a CNN
                    pitch_degree = my_head_pose_estimator.return_pitch(face,
                                                                    radians=False)  # Evaluate the pitch angle using a CNN
                    yaw_degree = my_head_pose_estimator.return_yaw(face, radians=False)  # Evaluate the yaw angle using a CNN
                    print("Estimated [roll, pitch, yaw] (degrees) ..... [" + str(roll_degree[0, 0, 0]) + "," + str(
                        pitch_degree[0, 0, 0]) + "," + str(yaw_degree[0, 0, 0]) + "]")
                    roll = my_head_pose_estimator.return_roll(face, radians=True)  # Evaluate the roll angle using a CNN
                    pitch = my_head_pose_estimator.return_pitch(face, radians=True)  # Evaluate the pitch angle using a CNN
                    yaw = my_head_pose_estimator.return_yaw(face, radians=True)  # Evaluate the yaw angle using a CNN
                    print("Estimated [roll, pitch, yaw] (radians) ..... [" + str(roll[0, 0, 0]) + "," + str(
                        pitch[0, 0, 0]) + "," + str(yaw[0, 0, 0]) + "]")
                    # Getting rotation and translation vector
                    rot_matrix = yaw2rotmat(
                        -yaw[0, 0, 0])  # Deepgaze use different convention for the Yaw, we have to use the minus sign

                    # Attention: OpenCV uses a right-handed coordinates system:
                    # Looking along optical axis of the camera, X goes right, Y goes downward and Z goes forward.
                    rvec, jacobian = cv2.Rodrigues(rot_matrix)
                    tvec = np.array([0.0, 0.0, 1.0], np.float)  # translation vector
                    print(rvec)

                    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)
                    p_start = (int(c_x), int(c_y))
                    p_stop = (int(imgpts[2][0][0]), int(imgpts[2][0][1]))
                    print("point start: " + str(p_start))
                    print("point stop: " + str(p_stop))
                    print("")

                    cv2.line(face, p_start, p_stop, (0, 0, 255), 3)  # RED
                    cv2.circle(face, p_start, 1, (0, 255, 0), 3)  # GREEN

                    yaw_degree = math.floor(yaw_degree[0, 0, 0])
                    print(yaw_degree)
                    direction = ""
                    if 45 <= yaw_degree <= 100:
                        print("The person is watching on the left")
                        direction = "left"
                    if -100 <= yaw_degree < -20:
                        print("The person is watching on the right")
                        direction = "right"
                    if -45 <= yaw_degree < 20:
                        print("The person is watching frontal")
                        direction = "frontal"

        else:
            direction = "back"

        watching = []
        for i, painting in enumerate(paintings_roi):
            x1, y1, x2, y2 = painting
            xpCenter = int((x1 + (x2 - x1) / 2))
            ypCenter = int((y1 + (y2 - y1) / 2))
            px, py = (xHead, yHead)
            if x1 < px or x2 < px and direction == 'left':
                cv2.circle(cp, (xpCenter, ypCenter), 1, person_gaze_lines_color, -1)
                #cv2.line(cp, (xHead, yHead), (xpCenter, ypCenter), person_gaze_lines_color, thickness=2)
                cv2.arrowedLine(cp, (xHead, yHead), (xpCenter, ypCenter), person_gaze_lines_color, thickness=2)
                if printTitlesOfPaintingsWatched:
                    watching.append(retrieval[i])
            if y1 < py or y2 < py and direction == 'back':
                cv2.circle(cp, (xpCenter, ypCenter), 1, person_gaze_lines_color, -1)
                #cv2.line(cp, (xHead, yHead), (xpCenter, ypCenter), person_gaze_lines_color, thickness=2)
                cv2.arrowedLine(cp, (xHead, yHead), (xpCenter, ypCenter), person_gaze_lines_color, thickness=2)
                if printTitlesOfPaintingsWatched:
                    watching.append(retrieval[i])
            if x1 > px or x2 > px and direction == 'right':
                cv2.circle(cp, (xpCenter, ypCenter), 1, person_gaze_lines_color, -1)
                #cv2.line(cp, (xHead, yHead), (xpCenter, ypCenter), person_gaze_lines_color, thickness=2)
                cv2.arrowedLine(cp, (xHead, yHead), (xpCenter, ypCenter), person_gaze_lines_color, thickness=2)
                if printTitlesOfPaintingsWatched:
                    watching.append(retrieval[i])

        if printTitlesOfPaintingsWatched:
            print("Person " + str(index) + " is watching:")
            for p in watching:
                print(p)

    return cp