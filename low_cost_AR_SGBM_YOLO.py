import asone
import numpy as np
from asone import utils
from asone import ASOne
from asone.utils import compute_color_for_labels
from asone.utils import get_names
import cv2 as cv
from functorch.dim import use_c
################################################################
global names
names =  get_names()

#############################################################
detector_L =ASOne(detector=asone.YOLOV7_PYTORCH, use_cuda=True)
detector_R =ASOne(detector=asone.YOLOV7_PYTORCH, use_cuda=True)

print('detector_L', detector_L)
print('detector_R', detector_R)

cap_L = cv.VideoCapture(0)
cap_R = cv.VideoCapture(2)

while cap_L.isOpened():
    ret_L, frame_L = cap_L.read()
    ret_R, frame_R = cap_R.read()

    detections_L, img_info_L = detector_L.detect(frame_L)
    detections_R, img_info_R = detector_R.detect(frame_R)

    bbox_xyxy_L = detections_L[:,:4]
    bbox_xyxy_R = detections_R[:,:4]
    scores_L = detections_L[:,4]
    scores_R = detections_R[:,4]
    class_ids_L = detections_L[:,5]
    class_ids_R = detections_R[:,5].astype(int)

    frame_L = utils.draw_boxes(frame_L, bbox_xyxy_L, scores_L, class_ids_L)
    frame_R = utils.draw_boxes(frame_R, bbox_xyxy_R, scores_R, class_ids_R)
    combined = np.concatenate((frame_L, frame_R), axis=1)
    cv.imshow("combined", combined)

    if cv.waitKey(1) & 0xFF == ord('q'):
        print(detections_L)
        print(bbox_xyxy_L)
        print(detections_L.shape)
        break


