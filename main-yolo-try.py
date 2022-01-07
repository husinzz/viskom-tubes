import cv2 as cv
import numpy as np
import sys


def rescaleFrame(frame, scale=0.75):  # Cool function
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


image = cv.imread(str(sys.argv[1]))  # Reads the image
# Rescales the image so it wont take the whole screen when shown
image = rescaleFrame(image, float(sys.argv[2]))

classes = None
with open('./yolo/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Init the neaural network ( yes neaural network )
network = cv.dnn.readNet('./yolo/yolov3.weights', './yolo/yolov3.cfg')
network.setInput(cv.dnn.blobFromImage(
    image, 0.00392, (416, 416), (0, 0, 0), True, crop=False))

layerNames = network.getLayerNames()
outputLayers = [layerNames[i[0] - 1]
                for i in network.getUnconnectedOutLayers()]
outs = network.forward(outputLayers)

class_ids = []
confidences = []
boxes = []

count = 0
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.1:
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            w = int(detection[2] * image.shape[1])
            h = int(detection[3] * image.shape[0])
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])


indices = cv.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)

for i in indices:
    i = i[0]
    box = boxes[i]
    if class_ids[i] == 0:
        count = count + 1
        label = str(classes[class_id])
        cv.rectangle(image, (round(box[0]), round(box[1])), (round(
            box[0]+box[2]), round(box[1]+box[3])), (0, 0, 0), 2)
        cv.putText(image, label, (round(
            box[0])-10, round(box[1])-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

print("Counted : ", count)
cv.putText(image, r"Amount of people : {}".format(count), (0,30), cv.FONT_HERSHEY_SIMPLEX, .8, (0,255,0), 2)
cv.imshow("img", image)
cv.waitKey(10000)
