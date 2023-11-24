import cv2

# Joint numbers: 0 for the head, 1 for the neck, etc.
BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
              "Background": 15}

# Pairs when connecting joints with lines
POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
              ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
              ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

# List of pre-trained models and their corresponding files
protoFile = "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\body_25\\pose_deploy.prototxt"
models = {
    "MPI": {
        "proto": "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\mpi\\pose_deploy_linevec_faster_4_stages.prototxt",
        "weights": "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\mpi\\pose_iter_160000.caffemodel"
    },
    "COCO": {
        "proto": "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\coco\\pose_deploy_linevec.prototxt",
        "weights": "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\coco\\pose_iter_440000.caffemodel"
    },
    "BODY_25": {
        "proto": "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\body_25\\pose_deploy.prototxt",
        "weights": "C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\openpose-master\\openpose-master\\models\\pose\\body_25\\pose_iter_584000.caffemodel"
    }
}

# Select the desired model
model_name = "BODY_25"

# Set up the trained network
protoFile = models[model_name]["proto"]
weightsFile = models[model_name]["weights"]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Read the test image
image = cv2.imread("C:\\Users\\82109\\Downloads\\Machine_Learning_lecture\\soft_lecture\\aistudent.jpg")

# Get height, width, and color information from the test image
imageHeight, imageWidth, imageColor = image.shape

# Preprocess the test image to put it into the network
inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

# Put the test image into the network
net.setInput(inpBlob)

# Get the result
output = net.forward()

H = output.shape[2]
W = output.shape[3]

# Draw the detected joint points on the test image
points = []
for i in range(0, 15):
    probMap = output[0, i, :, :]
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    x = (imageWidth * point[0]) / W
    y = (imageHeight * point[1]) / H
    if prob > 0.1:
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1)
        cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        points.append((int(x), int(y)))
    else:
        points.append(None)

cv2.imshow("Output-Keypoints", image)
cv2.waitKey(0)

# Connect joints with lines
for pair in POSE_PAIRS:
    partA = pair[0]
    partA = BODY_PARTS[partA]
    partB = pair[1]
    partB = BODY_PARTS[partB]

    if points[partA] and points[partB]:
        cv2.line(image, points[partA], points[partB], (255, 0, 0), 2)

cv2.imshow("Output-Keypoints-with-Lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
