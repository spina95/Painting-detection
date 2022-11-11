import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from paintings_busts_persons_detection import detect as detectPaintingsPersonsBusts
from detectron2_paintings_detection import detect_paintings
from painting_retrieval import PaintingFinder
import pytesseract as pt
import re
from PIL import Image
import argparse
from painting_rectification import rectify
import numpy as np
from argparse import RawTextHelpFormatter
import random
from people_watching_painting_detection import detect_people_watching_paintings_with_head_poze_estimetion
from swap_paintings_in_3d_model import swap_paintings_3d_model as exercise_optional_b


# def checkIfTemplateMatchingFoundSomething(threshold, tmRes):
#     for i in tmRes:
#         if i.any() > threshold:
#             return True
#     return False

def checkIfPersonsAreRealOrInPaintings(personsRois, paintingsRois):
    truePersons = []
    for personRoi in personsRois:
        personRoiHSV = cv2.cvtColor(personRoi, cv2.COLOR_BGR2HSV)
        found = 0
        h1 = cv2.calcHist([personRoiHSV], [0, 1], None, [180, 256], [0, 180, 0, 256])
        for paintingRoi in paintingsRois:
            paintingRoiHSV = cv2.cvtColor(paintingRoi, cv2.COLOR_BGR2HSV)
            #res = cv2.matchTemplate(paintingRoi, personRoi, cv2.TM_CCORR_NORMED)
            #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            #if max_val > 0.9:
            h2 = cv2.calcHist([paintingRoiHSV], [0, 1], None, [180, 256], [0, 180, 0, 256])
            res = cv2.compareHist(h1,h2, cv2.HISTCMP_CORREL)
            if res > 0.65:
                found = 1
                break
        if found == 0:
            truePersons.append(personRoi)
    return truePersons


def loadPaintingsDB(DBPath, CSVDBPath):
    db = pd.read_csv(CSVDBPath)
    paintingsDB = {}
    for imagefile in os.listdir(DBPath):
        tmpImg = cv2.imread(os.path.join(DBPath, imagefile))
        tmpImgRGB = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2RGB)
        tmpImgGray = cv2.cvtColor(tmpImgRGB, cv2.COLOR_RGB2GRAY)
        info = db[db['Image']==imagefile]
        paintingsDB[imagefile] = (tmpImg, tmpImgRGB, tmpImgGray,info['Title'].values.item(),info['Author'].values.item(),info['Room'].values.item())
    return paintingsDB

def mplot2cvimage(plot):
    plot.canvas.draw()
    data = np.fromstring(plot.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(plot.canvas.get_width_height()[::-1] + (3,))
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    return data

def obtainRoomsNumberLocationFromMap(mapImage):
    tmpImgGray = cv2.cvtColor(mapImage, cv2.COLOR_BGR2GRAY)
    thresh_binary = cv2.threshold(tmpImgGray, 0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 14))
    h_detected_lines = cv2.morphologyEx(thresh_binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    v_detected_lines = cv2.morphologyEx(thresh_binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    hcnts = cv2.findContours(h_detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vcnts = cv2.findContours(v_detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hcnts = hcnts[0] if len(hcnts) == 2 else hcnts[1]
    vcnts = vcnts[0] if len(vcnts) == 2 else vcnts[1]

    for c in hcnts:
        cv2.drawContours(mapImage, [c], -1, (255, 255, 255), 9)
    for c in vcnts:
        cv2.drawContours(mapImage, [c], -1, (255, 255, 255), 9)
    thresh_binary_2 =  cv2.threshold(cv2.cvtColor(mapImage, cv2.COLOR_BGR2GRAY), 0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Find digits
    contours,hierarchy = cv2.findContours(thresh_binary_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roisArray = []
    cntToSkip = []

    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        foundFriend = 0
        skip = 0
        for cnt2 in cntToSkip:
            if cv2.boundingRect(cnt) == cv2.boundingRect(cnt2):
                skip = 1
                break
        if skip == 1:
            continue

        for cnt1 in contours:
            skip1 = 0
            for cnt3 in cntToSkip:
                if cv2.boundingRect(cnt1) == cv2.boundingRect(cnt3):
                    skip1 = 1
                    break
            if skip1 == 1:
                continue

            if cv2.boundingRect(cnt) == cv2.boundingRect(cnt1):
                continue
            [x1, y1, w1, h1] = cv2.boundingRect(cnt1)
            if abs(x-x1) < 30 and abs(y-y1) < 5:
                roisArray.append([x if x < x1 else x1, y, abs(x-x1)+ (w1 if w1 > w else w), h])
                cntToSkip.append(cnt1)
                foundFriend = 1
                break
        if foundFriend == 0:
            roisArray.append([x, y, w, h])
            cntToSkip.append(cnt)

    resultDict = {}
    pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    for roi in roisArray:
        mapImagePil = Image.fromarray(mapImage[roi[1]-3:roi[1]+roi[3]+3, roi[0]-3:roi[0]+roi[2]+3])
        text = pt.image_to_string(mapImagePil, config='--psm 6')
        text = re.sub('[^0-9]','', text)
        resultDict[text] = roi
        #cv2.rectangle(mapImage, (roi[0] - 3, roi[1] - 3), (roi[0] + roi[2] +3, roi[1] + roi[3] +3), (0, 0, 255), 1)

    #cv2.imshow('thresh', thresh_binary)
    #cv2.imshow('thresh 2', thresh_binary_2)
    #cv2.imshow('horizontal_detected_lines', h_detected_lines)
    #cv2.imshow('vertical_detected_lines', v_detected_lines)
    #cv2.imshow('image', mapImage)
    #cv2.waitKey()
    return resultDict

def getRandomRGBColor(seed):
    r = random.randint(seed, 255)
    g = random.randint(seed, 255)
    b = random.randint(seed, 255)
    rgb = [r, g, b]
    return rgb

def removePyplotPlotFrameAndAxis(plot_array):
    plot_array[0].get_xaxis().set_visible(False)
    plot_array[0].get_yaxis().set_visible(False)
    plot_array[1].get_xaxis().set_visible(False)
    plot_array[1].get_yaxis().set_visible(False)
    plot_array[2].get_xaxis().set_visible(False)
    plot_array[2].get_yaxis().set_visible(False)
    plot_array[0].spines['top'].set_visible(False)
    plot_array[0].spines['right'].set_visible(False)
    plot_array[0].spines['left'].set_visible(False)
    plot_array[1].spines['top'].set_visible(False)
    plot_array[1].spines['right'].set_visible(False)
    plot_array[1].spines['left'].set_visible(False)
    plot_array[2].spines['top'].set_visible(False)
    plot_array[2].spines['right'].set_visible(False)
    plot_array[2].spines['left'].set_visible(False)
    return plot_array

def create_and_save_video_of_yolo_detectron(videoPath, output_path, video_title, overwrite, frames_to_skip, show_video=False,  include_gaze_detection=False):

    frame_rate = frames_to_skip
    frame_count = 0
    fpath = os.path.join(output_path, video_title + "." + "avi")

    if os.path.isfile(fpath) and overwrite == False:
        print("File with path " + fpath + " already exits please specify another video title or change the output directory!")
        return

    cap = cv2.VideoCapture(videoPath)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        return
    video = None
    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_count += 1
        print("Frame: " + str(frame_count))
        output_frame = None
        if video is None:
            print("Output video path is " + fpath)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video = cv2.VideoWriter(fpath, fourcc, 30.0,(frame.shape[1],frame.shape[0]))

        if ret is True and frame_count % frame_rate == 0:
            frame_with_segmented_paintings, triple_paintings_boxes_masks = detect_paintings(frame, False)
            persons, busts, persons_rois, busts_rois, output_frame = detectPaintingsPersonsBusts(frame_with_segmented_paintings, show_img=False)

            if len(persons) > 0 and include_gaze_detection is True:
                paintings_roi = []
                for pprbbbbb in triple_paintings_boxes_masks:
                    paintings_roi.append(pprbbbbb[1])

                output_frame = detect_people_watching_paintings_with_head_poze_estimetion(output_frame, frame, persons_rois, paintings_roi,
                                                                           [], False)


            video.write(output_frame)
            if show_video:
                output_frame_r = cv2.resize(output_frame,
                                                     (int(output_frame.shape[1] * 0.6), int(output_frame.shape[0] * 0.6)),
                                                     interpolation=cv2.INTER_AREA)
                cv2.imshow('Video', output_frame_r)
                key = cv2.waitKey(5)
        elif not ret:
            break

    cap.release()
    video.release()
    # Closes all the opencv windows
    cv2.destroyAllWindows()

def exercise_one(videoPath, frame_rate=10):
    cap = cv2.VideoCapture(videoPath)
    # cap = cv2.VideoCapture('D:/Shared-Data/Materie-Magistrale/Vision And Cognitive Systems/Project/Project material/videos/000/VIRB0399.MP4')
    # cap = cv2.VideoCapture('D:/Shared-Data/Materie-Magistrale/Vision And Cognitive Systems/Project/Project material/videos/002/20180206_111817.MP4')
    # cap = cv2.VideoCapture('D:/Shared-Data/Materie-Magistrale/Vision And Cognitive Systems/Project/Project material/videos/002/20180206_114720.MP4')
    # cap = cv2.VideoCapture('D:/Shared-Data/Materie-Magistrale/Vision And Cognitive Systems/Project/Project material/videos/002/20180206_114604.MP4')
    # cap = cv2.VideoCapture('D:/Shared-Data/Materie-Magistrale/Vision And Cognitive Systems/Project/Project material/videos/003/GOPR1928.MP4')
    # cap = cv2.VideoCapture('D:/Shared-Data/Materie-Magistrale/Vision And Cognitive Systems/Project/Project material/videos/002/20180206_114506.MP4')
    # cap = cv2.VideoCapture('D:/Shared-Data/Materie-Magistrale/Vision And Cognitive Systems/Project/Project material/videos/002/20180206_113059.MP4')
    # cap = cv2.VideoCapture('D:/Shared-Data/Materie-Magistrale/Vision And Cognitive Systems/Project/Project material/videos/003/GOPR1924.MP4')
    # cap = cv2.VideoCapture('D:/Shared-Data/Materie-Magistrale/Vision And Cognitive Systems/Project/Project material/videos/005/GOPR2051.MP4')
    # cap = cv2.VideoCapture('D:/Shared-Data/Materie-Magistrale/Vision And Cognitive Systems/Project/Inside_Louvre_Museum_Paris.mp4')
    # cap = cv2.VideoCapture('D:/Shared-Data/Materie-Magistrale/Vision And Cognitive Systems/Project/Van_ghogh_museum/van_ghogh museum_tour_Trim01.mp4')
    frame_count = 0

    # Load the map and the paintings from database for faster comparison
    roomsCoordDict = {}
    mapImg = cv2.imread("project_utils/map.png")
    roomsCoordDict = obtainRoomsNumberLocationFromMap(mapImg.copy())
    # The paintings db is a dictionary that has per key the painting image filename
    # Each of the keys has a tuple as value which contains information about the painting
    # The elements of this tuple are (ImageInOpenCVFormatBGR, ImageInOpenCVFormatRGB, ImageInOpenCVFormatGray, Painting title, Painting author, Painting room)
    paintingsDB = loadPaintingsDB("paintings_db", "project_utils/data.csv")
    # Load image retraival class
    pf = PaintingFinder()
    mpl.rcParams['interactive'] == True

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_count += 1

        if ret is True and frame_count % frame_rate == 0:
            frame_with_segmented_paintings, triple_paintings_boxes_masks = detect_paintings(frame, False)
            persons, busts, persons_rois, busts_rois, frame_with_people_busts_and_segmented_paintings = detectPaintingsPersonsBusts(
                frame_with_segmented_paintings, show_img=False)
            frame_with_people_busts_and_segmented_paintings_r = cv2.resize(
                frame_with_people_busts_and_segmented_paintings, (
                int(frame_with_people_busts_and_segmented_paintings.shape[1] * 0.6),
                int(frame_with_people_busts_and_segmented_paintings.shape[0] * 0.6)), interpolation=cv2.INTER_AREA)
            cv2.imshow('Detection results', frame_with_people_busts_and_segmented_paintings_r)
            key = cv2.waitKey(5)
            Painting_PaintingBox_Rectified_BestMatchesList_BestMatch_BestMatchTitle_BestMatchAuthor_BestMatchRoom = []
            painting_and_rectified_painting_plot, painting_and_rectified_painting_plot_array = plt.subplots(1, 3)

            painting_and_rectified_painting_plot_array = removePyplotPlotFrameAndAxis(
                painting_and_rectified_painting_plot_array)

            # painting_and_rectified_painting_plot_array[1, 1].imshow(cv2.cvtColor(mapImgc, cv2.COLOR_BGR2RGB))
            # painting_and_rectified_painting_plot_array[1, 1].set_title('MAP')

            for tpbm in triple_paintings_boxes_masks:
                rectified, originalImagePointsForProspectiveTransformations, is_rect = rectify(frame, tpbm[2])
                retrievals = pf.getRankedListOfSimilars([rectified], paintingsDB, method=1)

                Painting_PaintingBox_Rectified_BestMatchesList_BestMatch_BestMatchTitle_BestMatchAuthor_BestMatchRoom.append(
                    (tpbm[0], tpbm[1], rectified, retrievals, paintingsDB[list(retrievals[0].keys())[0]][0],
                     paintingsDB[list(retrievals[0].keys())[0]][3], paintingsDB[list(retrievals[0].keys())[0]][4],
                     paintingsDB[list(retrievals[0].keys())[0]][5]))

                painting_and_rectified_painting_plot_array[0].imshow(cv2.cvtColor(tpbm[0], cv2.COLOR_BGR2RGB))
                painting_and_rectified_painting_plot_array[0].set_title('PAITNING')

                painting_and_rectified_painting_plot_array[1].imshow(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB))
                painting_and_rectified_painting_plot_array[1].set_title('RECTIFIED PAINTING')

                painting_and_rectified_painting_plot_array[2].imshow(
                    cv2.cvtColor(paintingsDB[list(retrievals[0].keys())[0]][0], cv2.COLOR_BGR2RGB))
                painting_and_rectified_painting_plot_array[2].set_title('BEST MATCH')

                plotImage = mplot2cvimage(painting_and_rectified_painting_plot)
                cv2.imshow('Painting - Rectified Painting - Best Match', plotImage)
                key = cv2.waitKey(5)

            # If there are persons in the frame then simply map them on the map
            if len(persons) > 0:
                # The room number is the room with the highest score within the rooms of best matched for each painting
                rooms = {}
                for pprbbbbb in Painting_PaintingBox_Rectified_BestMatchesList_BestMatch_BestMatchTitle_BestMatchAuthor_BestMatchRoom:
                    if pprbbbbb[7] not in rooms:
                        rooms[pprbbbbb[7]] = 1
                    else:
                        rooms[pprbbbbb[7]] += 1
                rooms = {k: v for k, v in sorted(rooms.items(), key=lambda item: item[1], reverse=True)}
                room = list(rooms.keys())[0]
                print("Found " + str(len(persons)) + " people in room " + str(room))

                # Show the map with different colors points as indicating people on it
                mapImgc = mapImg.copy()
                # Scale down the map to better integrate it into the pyplot
                mapScaleFactor = 0.6

                newH = plotImage.shape[0]
                newW = plotImage.shape[1]

                nMapH = int(mapImgc.shape[0] * mapScaleFactor)
                nMapW = int(mapImgc.shape[1] * mapScaleFactor)
                mapImgc = cv2.resize(mapImgc, (nMapW, nMapH), interpolation=cv2.INTER_AREA)

                # Show the map with different colors points as indicating people on it
                mapImgc = mapImg.copy()

                pointThikness = int(8 * mapScaleFactor)
                xRoomCord = int((roomsCoordDict[str(room)][0] * mapScaleFactor) + pointThikness)
                yRoomCord = int((roomsCoordDict[str(room)][0] * mapScaleFactor) + pointThikness)
                mapImgc = cv2.resize(mapImgc, (nMapW, nMapH), interpolation=cv2.INTER_AREA)
                dictPersonCoordOnMap = {}
                for i in range(len(persons)):
                    pY = random.randint(yRoomCord, yRoomCord + 50)
                    pX = random.randint(xRoomCord, xRoomCord + 50)
                    while (pX, pY) in dictPersonCoordOnMap.values():
                        pY = random.randint(yRoomCord, 50)
                        pX = random.randint(xRoomCord, 50)
                    dictPersonCoordOnMap[i] = (pX, pY)
                    cv2.circle(mapImgc, (pX, pY), pointThikness, getRandomRGBColor(i), -1)

                # painting_and_rectified_painting_plot_array[1, 1].imshow(cv2.cvtColor(mapImgc, cv2.COLOR_BGR2RGB))
                # painting_and_rectified_painting_plot_array[1, 1].set_title('MAP')
                # painting_and_rectified_painting_plot_array[1, 1].get_xaxis().set_visible(False)
                # painting_and_rectified_painting_plot_array[1, 1].get_yaxis().set_visible(False)

                # pr = mplot2cvimage(painting_and_rectified_painting_plot)
                cv2.imshow('Map - Persons will be drawn as points on this map', mapImgc)

                # Other then showing persons on the map we analyze there gaze and determine at which painting they are looking at
                paintings_roi = []
                best_matches_title = []
                for pprbbbbb in Painting_PaintingBox_Rectified_BestMatchesList_BestMatch_BestMatchTitle_BestMatchAuthor_BestMatchRoom:
                    paintings_roi.append(pprbbbbb[1])
                    best_matches_title.append(pprbbbbb[5])

                frame_with_gaze_lines = detect_people_watching_paintings_with_head_poze_estimetion(frame_with_people_busts_and_segmented_paintings, frame, persons_rois, paintings_roi,
                                                                           best_matches_title)

                frame_resize_factor = 0.6
                frame_with_gaze_lines_ = cv2.resize(frame_with_gaze_lines, (int(frame_with_gaze_lines.shape[1] * frame_resize_factor), int(frame_with_gaze_lines.shape[0] * frame_resize_factor)),
                                interpolation=cv2.INTER_AREA)
                cv2.imshow("Persons looking at paintings", frame_with_gaze_lines_)

            key = cv2.waitKey(5)
            # while key not in [ord('q'), ord('k')]:
            #key = cv2.waitKey(0)
        elif not ret:
            break

    cap.release()

    # Closes all the opencv windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test', formatter_class=RawTextHelpFormatter)
    parser.add_argument("exercise_number", choices=[0,1,2], help = "Possible values of exercise_number are: [0,1] \n 0 = Show projects exercises A,B,C,D,E,Optional_A \n 1 = Show projects exercises Optional_B \n 2 Save a video of Yolo and Detectron2 in action", type=int)
    parser.add_argument("-vp", "--video_path", help = "A path to a video file to use for exercise_number=0")
    parser.add_argument("-sv", "--show_video", help = "Show the video while making it [0=False 1=True]", action="store_true")
    parser.add_argument("-igl", "--include_gaze_lines", help = "Include gaze lines in the output video" , action="store_true")
    parser.add_argument("-oe", "--overwrite_existing", help = "Overwrite existing file [0=False 1=True]", action="store_true")
    parser.add_argument("-fts", "--frames_to_skip", help = "Specify a number of frames to skip [0,...,N]", type=int)
    parser.add_argument("-ovdp", "--output_video_directory_path", help = "Specify the directory path of output video")
    parser.add_argument("-ovt", "--output_video_title", help = "Specify title of the output video")
    args = parser.parse_args()

    if args.exercise_number == 0:

        if args.video_path is None:
            print("A path to a video file to use is required!")
            exit(1)
        if not os.path.isfile(args.video_path):
            print(str(args.video_path) + "is not a valid video path!")
            exit(1)

        exercise_one(args.video_path, 10 if args.frames_to_skip is None else args.frames_to_skip)

    elif args.exercise_number == 1:
        exercise_optional_b()
    elif args.exercise_number == 2:
        if args.video_path is None:
            print("A path to a video file to use is required!")
            exit(1)

        if not os.path.isfile(args.video_path):
            print(str(args.video_path) + "is not a valid video path!")
            exit(1)
        if args.output_video_directory_path is None or args.output_video_directory_path == "":
            print("Argument output_video_directory_path is required!")
            exit(1)
        if args.output_video_title is None or args.output_video_title == "":
            print("Argument output_video_title is required!")
            exit(1)

        create_and_save_video_of_yolo_detectron(args.video_path, args.output_video_directory_path, args.output_video_title, args.overwrite_existing, 1 if args.frames_to_skip is None else args.frames_to_skip, args.show_video,  args.include_gaze_lines)

    #exercise_one('D:/Shared-Data/Materie-Magistrale/Vision And Cognitive Systems/Project/Project material/videos/002/20180206_114604.MP4', 20)


