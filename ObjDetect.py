import cv2
thres = 0.5  # Threshold to detect object

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, boundbox = net.detect(img, confThreshold=thres)
    print(classIds, boundbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), boundbox):
            cv2.rectangle(img, box, color=(229, 250, 5), thickness=2)
            h = box[3] - box[1]
            w = box[2] - box[0]
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (189, 5, 250), 2)
            cv2.putText(img, f'{round(confidence*100)}% sure', (box[0] + 200, box[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (5, 99, 250), 2)

    cv2.imshow("Object detector", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
