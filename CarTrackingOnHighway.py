# This python script is for tracking objects on highway, we have considered a video feed from highway
# Purpose of this program is to keep the track of vehicles and assigning unique id to each unique vehicle spotted
# I've not used any algorithm for object tracking, hence this is implemented from scratch.

import cv2
from object_detection import ObjectDetection # Importing yolo which is available in another python file
import math

Object_Detection = ObjectDetection()
cap = cv2.VideoCapture("los_angeles.mp4")
count = 0
center_point_of_prevFrame = []
tracking_objects = {}
tracking_id = 0

while( True ):
    count += 1
    center_point_of_currFrame = []
    # ret will true when we have frames and false when no frames.
    ret, frame = cap.read()
    if not ret:
        break
    # capturing 1 frame at a time and its information then
    # drawing rectagle when show the video..i.e detecting objects
    (class_id, scores, bounding_box) = Object_Detection.detect(frame)

    # drawing bounding boxes around all objects in each frame
    for box in bounding_box:
        (x, y, w, h) = box
        centre_x = int((x + x+w)/2)  # x coordinate has to be int
        centre_y = int((y + y+h)/2)  # y coordinate has to be int
        center_point_of_currFrame.append((centre_x,centre_y)) # adding centre points for tracking
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0), 2)

    # Tracking
    if count <= 2:
        for curr_point in center_point_of_currFrame:
            for prev_point in center_point_of_prevFrame:
                distance = math.dist(curr_point, prev_point)

                if distance < 20: # if its below certain threshold its the same hold.
                    tracking_objects[tracking_id] = curr_point
                    tracking_id += 1
    else:
        # Keeping copies as we cannot loop and remove elements at the same time.
        tracking_objects_copy = tracking_objects.copy()
        center_point_of_currFrame_copy = center_point_of_currFrame.copy()

        for ob_id, obj_coordinate in tracking_objects_copy.items():
            is_object_exist_inFrame = False
            for curr_point in center_point_of_currFrame_copy:
                distance = math.dist(curr_point, obj_coordinate)

                if distance < 20:  # implies that in previous frame and current frame same car is found so add same id and point
                    is_object_exist_inFrame = True
                    tracking_objects[ob_id] = curr_point  # updating the new position of the object found..

                    if curr_point in center_point_of_currFrame:
                        center_point_of_currFrame.remove(curr_point)
                    continue  # skipping as we have already found the object no point of searching in entire list

            if not is_object_exist_inFrame: # If object does not exist i.e gone in frame we remove it from dictionary
                tracking_objects.pop(ob_id)

        # Adding new ids to new objects that we are seeing..
        for point in center_point_of_currFrame:
            tracking_objects[tracking_id] = point
            tracking_id += 1

    # Drawings center points and Labeling ids for recognition of tracked objects
    for ob_id, obj_coordinate in tracking_objects.items():
        cv2.circle(frame, obj_coordinate, 3, (0,0,255),-1)
        cv2.putText(frame,str(ob_id),( obj_coordinate[0], obj_coordinate[1]-7),fontFace=0,fontScale=1,color=(0,0,255))

    print("Tracking Objects: ", tracking_objects)
    print("Current frame center points left:", center_point_of_currFrame)

    # waiting for 1 sec for each frame and we display and it also waits for keestrock
    # when key "S" is pressed it will record in key strock and we will exist i.e if we want to stop
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

    # storing current points so that we can use in next iter for comparision for id assignment
    center_point_of_prevFrame = center_point_of_currFrame.copy()

cap.release()
cv2.destroyAllWindows()