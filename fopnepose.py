################### https://bskyvision.com/1164 ##############
import cv2

# 관절 번호: 머리는 0, 목은 1 등등
BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,   ##머리, 목, 어깨, 팔꿈치, 손목
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,  ##어깨, 팔꿈치, 손목, 엉덩이, 무릎
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14, ##발목, 엉덩이, 무릎, 발목, 가슴
              "Background": 15}                                                 ##등
 
# 관절들을 선으로 이을 때 쌍이 되는 것들
POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
              ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
              ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]
 
# 훈련된 network 세팅
# protoFile = "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\mpi\\pose_deploy_linevec_faster_4_stages.prototxt"
# weightsFile = "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\mpi\\pose_iter_160000.caffemodel" ##mpii


# protoFile = "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\coco\\pose_deploy_linevec.prototxt"
# weightsFile = "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\coco\\pose_iter_440000.caffemodel"   ##coco


protoFile = "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\body_25\\pose_deploy.prototxt"
weightsFile = "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\body_25\\pose_iter_584000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
 
# 테스트 이미지 읽기
image = cv2.imread("C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\aistudent.jpg")
 
# 테스트 이미지에서 height, width, color 정보 파악
imageHeight, imageWidth, imageColor = image.shape
 
# 테스트 이미지를 network에 넣기 위해 전처리
inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
 
# 테스트 이미지를 network에 넣어줌
net.setInput(inpBlob)
 
# 결과 받아오기
output = net.forward()
 
H = output.shape[2]
W = output.shape[3]
 
# 검출된 관절 포인트를 테스트 이미지에 그려주기
points = []
for i in range(0, 15):
    # 해당 관절 신뢰도 얻기
    probMap = output[0, i, :, :]
 
    # global 최대값 찾기
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
 
    # 원래 이미지에 맞게 점 위치 변경
    x = (imageWidth * point[0]) / W
    y = (imageHeight * point[1]) / H
 
    # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로
    if prob > 0.1:
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1,
                   lineType=cv2.FILLED)  # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                    lineType=cv2.LINE_AA)
        points.append((int(x), int(y)))
    else:
        points.append(None)
 
cv2.imshow("Output-Keypoints", image)
cv2.waitKey(0)
 
# 관절들을 선으로 연결해주기
for pair in POSE_PAIRS:
    partA = pair[0]  # Head
    partA = BODY_PARTS[partA]  # 0
    partB = pair[1]  # Neck
    partB = BODY_PARTS[partB]  # 1
 
    # print(partA," 와 ", partB, " 연결\n")
    if points[partA] and points[partB]:
        cv2.line(image, points[partA], points[partB], (255, 0, 0), 2)
 
cv2.imshow("Output-Keypoints-with-Lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
