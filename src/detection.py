# C�digo para usar YOLO en opencv
# Docu: https://blog.francium.tech/custom-object-training-and-detection-with-yolov3-darknet-and-opencv-41542f2ff44e

import argparse
import cv2
import numpy as np
import statistics

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--image", default='media/test.png', help="image for prediction")
parser.add_argument("--video", default='media/nude_cones_hot_videos/Webcam/3m_static.webm', help="class names path")
parser.add_argument("--config", default='cfg/yolov3-tiny-UPMR.cfg', help="YOLO config path")
parser.add_argument("--weights", default='weights/yolov3-tiny-UPMR.weights', help="YOLO weights path")
parser.add_argument("--names", default='data/UPMR.names', help="class names path")
args = parser.parse_args()

CONF_THRESH, NMS_THRESH = 0.5, 0.5

# Load the network
net = cv2.dnn.readNetFromDarknet(args.config, args.weights)

# Para usar CUDA o no 

#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Get the output layer from YOLO
layers = net.getLayerNames()
output_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]

# Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
img = cv2.imread(args.image)

# Open the video file
video = cv2.VideoCapture(args.video)

# Check if the video file was successfully opened
if not video.isOpened():
    print("Error opening video file.")

height, width = img.shape[:2]

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layer_outputs = net.forward(output_layers)

class_ids, confidences, b_boxes = [], [], []
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONF_THRESH:
            center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            b_boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))

# Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()

# Draw the filtered bounding boxes with their class to the image
with open(args.names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Diccionario para que saque la clase 0 en amarillo; clase 1 en azul y clase 2 en naranja
colordict = {
  0: (204, 255, 255),
  1: (255, 0, 0),
  2: (0, 191, 255)
}

for index in indices:
    x, y, w, h = b_boxes[index]




# Margin for each bounding box
margin = 0.2
triangle_top_points = []
trapeze_bot_left_points = []
trapeze_bot_right_points = []
trapeze_top_left_points = []
trapeze_top_right_points = []
triangle_left_points = []
triangle_right_points = []

for b_box  in b_boxes:
    x, y, w, h = b_box

    # Calculate the expanded region coordinates
    x_min = max(0, x - int(w*margin))
    y_min = max(0, y - int(h*margin))
    x_max = min(img.shape[1] - 1, x + w + int(w*margin))
    y_max = min(img.shape[0] - 1, y + h + int(h*margin))

    # Extract the region of interest (ROI) from the original image
    roi = img[y_min:y_max, x_min:x_max]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)        # We cast to HSV: Hue, Saturation, Value. It is better since "color" (hue) is continuous, unlike in RGB.

    lower_blue = np.array([113, 134, 9])                 # We define the color ranges
    upper_blue = np.array([129, 255, 44])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)       # We create a mask with the color range we defined
    res = cv2.bitwise_and(roi, roi, mask=mask)      # Logical and between mask and frame

    # Find contours in the eroded image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort contours by area
    contours = contours[:2] # Keep the 2 biggest ones

    # Draw the contour of the cone
    # cv2.drawContours(roi, contours, -1, (0,255,0))

    
    # Get the bigger one (bottom trapeze)
    max_contour = contours[0]
    # Get the smaller one (top triangle)
    min_contour = contours[1]
    

    # Get top point of the triangle
    tri_top_point = min(min_contour, key=lambda point: point[0][1])[0]
    # Get left point of the triangle
    tri_left_point = min(min_contour, key=lambda point: point[0][0])[0]
    # Get top point of the triangle
    tri_right_point = max(min_contour, key=lambda point: point[0][0])[0]

    # Get left point of the trapeze
    tra_left_point = min(max_contour, key=lambda point: point[0][0])[0]
    # Get the right point of the trapeze
    tra_right_point = max(max_contour, key=lambda point: point[0][0])[0]
    # Get top left point of the trapeze
    tra_topleft_point = max(max_contour, key=lambda point: -71 * point[0][0] - 100 * point[0][1])[0]
    # Get top right point of the trapeze
    tra_topright_point = max(max_contour, key=lambda point: 71 * point[0][0] - 100 * point[0][1] )[0]

    # Transform to global coordinates and append the points
    triangle_top_points.append([x_min + tri_top_point[0], y_min + tri_top_point[1]])
    triangle_left_points.append([x_min + tri_left_point[0], y_min + tri_left_point[1]])
    triangle_right_points.append([x_min + tri_right_point[0], y_min + tri_right_point[1]])

    trapeze_bot_left_points.append([x_min + tra_left_point[0], y_min + tra_left_point[1]])
    trapeze_bot_right_points.append([x_min + tra_right_point[0], y_min + tra_right_point[1]])
    trapeze_top_left_points.append([x_min + tra_topleft_point[0], y_min + tra_topleft_point[1]])
    trapeze_top_right_points.append([x_min + tra_topright_point[0], y_min + tra_topright_point[1]])

    # Draw the circles on the image
    cv2.circle(roi, (tri_top_point[0], tri_top_point[1]), 5, (0, 0, 255), -1)  # -1 fills the circle
    cv2.circle(roi, (tri_left_point[0], tri_left_point[1]), 5, (0, 0, 255), -1)  # -1 fills the circle
    cv2.circle(roi, (tri_right_point[0], tri_right_point[1]), 5, (0, 0, 255), -1)  # -1 fills the circle

    cv2.circle(roi, (tra_right_point[0], tra_right_point[1]), 5, (0, 0, 255), -1)  # -1 fills the circle
    cv2.circle(roi, (tra_left_point[0], tra_left_point[1]), 5, (0, 0, 255), -1)  # -1 fills the circle
    cv2.circle(roi, (tra_topright_point[0], tra_topright_point[1]), 5, (0, 0, 255), -1)  # -1 fills the circle
    cv2.circle(roi, (tra_topleft_point[0], tra_topleft_point[1]), 5, (0, 0, 255), -1)  # -1 fills the circle

# Calibrate these parameters
camera_baseline = 0.119953  # Baseline in m
camera_focal = 466  # De momento es invent

# Compute disparities
disparity_tri_top = (triangle_top_points[0][0] - width/4) - (triangle_top_points[1][0] - 3*width/4)
disparity_tri_right = (triangle_right_points[0][0] - width/4) - (triangle_right_points[1][0] - 3*width/4)
disparity_tri_left = (triangle_left_points[0][0] - width/4) - (triangle_left_points[1][0] - 3*width/4)


disparity_trap_left = (trapeze_bot_left_points[0][0] - width/4) - (trapeze_bot_left_points[1][0] - 3*width/4)
disparity_trap_right = (trapeze_bot_right_points[0][0] - width/4) - (trapeze_bot_right_points[1][0] - 3*width/4)
disparity_trap_topleft = (trapeze_top_left_points[0][0] - width/4) - (trapeze_top_left_points[1][0] - 3*width/4)
disparity_trap_topright = (trapeze_top_right_points[0][0] - width/4) - (trapeze_top_right_points[1][0] - 3*width/4)

# Compute depths
depth_tri_top = camera_focal*camera_baseline/disparity_tri_top
depth_tri_right = camera_focal*camera_baseline/disparity_tri_right
depth_tri_left = camera_focal*camera_baseline/disparity_tri_left


depth_trap_left = camera_focal*camera_baseline/disparity_trap_left
depth_trap_right = camera_focal*camera_baseline/disparity_trap_right
depth_trap_topleft = camera_focal*camera_baseline/disparity_trap_topleft
depth_trap_topright = camera_focal*camera_baseline/disparity_trap_topright

# Statistical parameters
mean_depth = statistics.mean([depth_tri_top, depth_tri_right, depth_tri_left, depth_trap_left, depth_trap_right, depth_trap_topleft, depth_trap_topright])
stddev_depth = statistics.stdev([depth_tri_top, depth_tri_right, depth_tri_left, depth_trap_left, depth_trap_right, depth_trap_topleft, depth_trap_topright])

# Draw lines between the points
cv2.line(img, triangle_top_points[0], triangle_top_points[1], (255, 255, 255), 1)  # Adjust thickness as desired
cv2.line(img, triangle_left_points[0], triangle_left_points[1], (255, 255, 255), 1) 
cv2.line(img, triangle_right_points[0], triangle_right_points[1], (255, 255, 255), 1) 

cv2.line(img, trapeze_bot_left_points[0], trapeze_bot_left_points[1], (255, 255, 255), 1) 
cv2.line(img, trapeze_bot_right_points[0], trapeze_bot_right_points[1], (255, 255, 255), 1)  
cv2.line(img, trapeze_top_left_points[0], trapeze_top_left_points[1], (255, 255, 255), 1)  
cv2.line(img, trapeze_top_right_points[0], trapeze_top_right_points[1], (255, 255, 255), 1)  

 
cv2.putText(img, "Average depth: {:.3}m".format(mean_depth), (width//2 - 500, height - 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
cv2.putText(img, "Std deviation: {:.3}m".format(stddev_depth), (width//2 - 500, height - 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

# 3D position based on computed depth
#   X = Z / fx * (u - cx)
#   Y = Z / fy * (v - cy)
projected_center = [statistics.mean([triangle_top_points[0][0], triangle_left_points[0][0], triangle_right_points[0][0], trapeze_bot_left_points[0][0], trapeze_bot_right_points[0][0], trapeze_top_left_points[0][0], trapeze_top_right_points[0][0]]), statistics.mean([triangle_top_points[0][1], triangle_left_points[0][1], triangle_right_points[0][1], trapeze_bot_left_points[0][1], trapeze_bot_right_points[0][1], trapeze_top_left_points[0][1], trapeze_top_right_points[0][1]])]

# Draw circle in the middle of the cone
cv2.circle(img, (projected_center[0], projected_center[1]), 5, (255, 255, 0), -1)

x_pos = mean_depth / camera_focal * (projected_center[0] - width/4)
y_pos = mean_depth / camera_focal * (projected_center[1] - height/2)

cv2.putText(img, "Computed position: ({:.3},{:.3},{:.3})m".format(x_pos, y_pos, mean_depth), (width//2 - 500, height - 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
