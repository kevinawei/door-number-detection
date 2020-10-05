import cv2
import os
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import digit_detector.region_proposal as rp
import digit_detector.show as show
import digit_detector.detect as detector
import digit_detector.file_io as file_io
import digit_detector.preprocess as preproc
import digit_detector.classify as cls
import pp as pre
import argparse
from imutils.object_detection import non_max_suppression
from cv2 import VideoWriter, VideoWriter_fourcc

format = "XVID"

def transform(box, crop_set):
    y1, y2, x1, x2 = box
    x0,y0,w,h = crop_set
    xl = x1 + x0
    xr = xl + x2 - x1
    yl = y1 + y0
    yr = yl + y2 - y1
    return [xl,xr,yl,yr]


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type = str, help = "input video file path")
args = vars(ap.parse_args())
detector_model = "detector_model.hdf5"
recognize_model = "recognize_model.hdf5"

mean_value_for_detector = 107.524
mean_value_for_recognizer = 112.833

model_input_shape = (32,32,1)
print(args)
vs = cv2.VideoCapture(args["video"])
filename = input("Please input Filename: ")

fps = FPS().start()
preproc_for_detector = preproc.GrayImgPreprocessor(mean_value_for_detector)
preproc_for_recognizer = preproc.GrayImgPreprocessor(mean_value_for_recognizer)

char_detector1 = cls.CnnClassifier(detector_model, preproc_for_detector, model_input_shape)
char_recognizer = cls.CnnClassifier(recognize_model, preproc_for_recognizer, model_input_shape)

#.........................
frameID = 1
fOut = ""
file = open(filename + '.txt', 'w')
#.........................
digit_spotter = detector.DigitSpotter(char_detector1, char_recognizer, rp.MserRegionProposer())
orig_list = []

while True:
    # grab the current frame
    frame = vs.read()
    frame = frame[1]
    # check to see if we have reached the end of the stream
    if frame is None:
        break

    crops = pre.preprocess(frame)
    for crop, crop_set in crops:
        output= digit_spotter.run(crop, threshold=0.5, do_nms=True,show_result=True, nms_threshold=0.1)
        if output is None:
            fOut = "No Digits Detected"
            file.write("Frame " + str(frameID) + ': ' + fOut + "\n")
            continue

        for box,prob,msg in output:
            box =  transform (box,crop_set)
            cv2.rectangle(frame, (box[0],box[2]),(box[1],box[3]),(0, 0, 225), 2)
            cv2.putText(frame, msg, (box[0], box[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
            fOut = str(msg)
            file.write(
                "Frame " + str(frameID) + ': ' + '[' + str(box[0]) + ', ' + str(box[2]) + ', ' + str(
                    box[1]) + ', ' + str(
                    box[3]) + '], ' + fOut + "\n")
    frameID += 1
    fps.update()

fps.stop()
vs.release()
print("Detection Complete")
