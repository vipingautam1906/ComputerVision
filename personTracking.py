from object_detection import ObjectDetection
import cv2
import math

od = ObjectDetection()
webcam = cv2.VideoCapture(0)
frame_count = 0

while True:
    status, frame = webcam.read(0)
    if status is False:
        break
    frame_count += 1

    class_id, class_prob, boundingBoxes = od.detect(frame)
    for bbox in boundingBoxes:
        x, y, w, h = bbox
        cv2.rectangle(frame,(x,y),(x+w, y+h),(200,255,0),2)
    cv2.imshow('Output', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
webcam.release()
cv2.destroyAllWindows()