import cv2
import numpy as np

# YOLO 모델 로드
net_yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names_yolo = net_yolo.getLayerNames()
output_layers_yolo = [layer_names_yolo[i[0] - 1] for i in net_yolo.getUnconnectedOutLayers()]


# 훈련된 network 세팅
# protoFile = "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\mpi\\pose_deploy_linevec_faster_4_stages.prototxt"
# weightsFile = "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\mpi\\pose_iter_160000.caffemodel" ##mpii

protoFile = "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\coco\\pose_deploy_linevec.prototxt"
weightsFile = "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\coco\\pose_iter_440000.caffemodel"   ##coco

# protoFile = "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\body_25\\pose_deploy.prototxt"
# weightsFile = "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\body_25\\pose_iter_584000.caffemodel"

# OpenPose 모델 로드
net_openpose = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# YOLO을 위한 클래스 레이블 로드
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 이미지 로드
image = cv2.imread("C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\aistudent.jpg")
height, width, _ = image.shape

# YOLO을 사용하여 객체 감지
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net_yolo.setInput(blob)
outs_yolo = net_yolo.forward(output_layers_yolo)

class_ids = []
confidences = []
boxes = []

for out in outs_yolo:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# 중복된 겹치는 박스를 제거하기 위해 비최대 억제(NMS) 적용
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# 이미지에 경계 상자와 클래스 레이블 그리기
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# OpenPose를 위해 이미지 전처리
blob_openpose = cv2.dnn.blobFromImage(image, 1.0 / 255, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
net_openpose.setInput(blob_openpose)

# OpenPose를 사용하여 포즈 추정
output_openpose = net_openpose.forward()

# 추정된 포즈를 이미지에 그리기
H = output_openpose.shape[2]
W = output_openpose.shape[3]
points = []

for i in range(18):
    prob_map = output_openpose[0, i, :, :]
    _, prob, _, point = cv2.minMaxLoc(prob_map)
    x = (width * point[0]) / W
    y = (height * point[1]) / H
    if prob > 0.1:
        cv2.circle(image, (int(x), int(y)), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(image, f"{i}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        points.append((int(x), int(y)))
    else:
        points.append(None)

for pair in [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (0, 14), (14, 16), (0, 15), (15, 17)]:
    part_a = pair[0]
    part_b = pair[1]
    if points[part_a] and points[part_b]:
        cv2.line(image, points[part_a], points[part_b], (0, 255, 255), 2)

# 결과 이미지 보여주기
cv2.imshow("YOLO + OpenPose", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
