import cv2
import numpy as np
import argparse
font = cv2.FONT_HERSHEY_COMPLEX

def nothing(self):
    pass
# cap = cv2.VideoCapture(0)   # For the main camera

parser = argparse.ArgumentParser(add_help=False)
# parser.add_argument("--image", default='media/test.png', help="image for prediction")
# parser.add_argument("--image", default="C:\\Users\\Juan\\Downloads\\iloveimg-resized (5)\\2m.jpg", help="image for prediction")
parser.add_argument("--image", default="media/test.png", help="image for prediction")
parser.add_argument("--calibration_file", default='cfg/hsvdata.txt', help="file to store results")
args = parser.parse_args()

# Calibration stuff
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 106, 180, nothing)
cv2.createTrackbar("L-S", "Trackbars", 38, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("W-H", "Trackbars", 133, 180, nothing)
cv2.createTrackbar("W-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("W-V", "Trackbars", 67, 255, nothing)

while (1):
    frame = cv2.imread(args.image)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)        # We cast to HSV: Hue, Saturation, Value. It is better since "color" (hue) is continuous, unlike in RGB.

    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    w_h = cv2.getTrackbarPos("W-H", "Trackbars")
    w_s = cv2.getTrackbarPos("W-S", "Trackbars")
    w_v = cv2.getTrackbarPos("W-V", "Trackbars")



    lower_blue = np.array([l_h, l_s, l_v])                 # We define the color ranges
    upper_blue = np.array([w_h, w_s, w_v])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)       # We create a mask with the color range we defined
    res = cv2.bitwise_and(frame, frame, mask=mask)      # Logical and between mask and frame

    res_hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    nonzero_indices = np.nonzero(np.any(res_hsv != [0, 0, 0], axis=2))
    non_zero_values = res_hsv[nonzero_indices]
    mean_hsv = np.mean(non_zero_values, axis=0)
    std_hsv = np.std(non_zero_values, axis=0)

    low95hsv = mean_hsv-1.645*std_hsv
    high95hsv = mean_hsv+1.645*std_hsv

    low95hsv[low95hsv < 0] = 0
    high95hsv[high95hsv > 255] = 255
    if(high95hsv[0] > 179):
        high95hsv[0] = 179
    


    cv2.putText(res, "Average hsv: [{:.2f},{:.2f},{:.2f}]".format(mean_hsv[0], mean_hsv[1], mean_hsv[2]), (100, res.shape[0] - 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    cv2.putText(res, "Standard deviation: [{:.2f},{:.2f},{:.2f}]".format(std_hsv[0], std_hsv[1], std_hsv[2]), (100, res.shape[0] - 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    cv2.putText(res, "HSV interval for 90% of masked points: [{:.0f}-{:.0f}, {:.0f}-{:.0f}, {:.0f}-{:.0f}]".format(low95hsv[0], high95hsv[0], low95hsv[1], high95hsv[1], low95hsv[2], high95hsv[2]), (100, res.shape[0] - 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    # check esc key to exit
    k = cv2.waitKey(5) & 0xff
    if(k == ord('s') or k == ord('S')):
        with open(args.calibration_file, "w") as file:
            file.write("Average hsv: [{:.2f},{:.2f},{:.2f}]\n".format(mean_hsv[0], mean_hsv[1], mean_hsv[2]))
            file.write("Standard deviation: [{:.2f},{:.2f},{:.2f}]\n".format(std_hsv[0], std_hsv[1], std_hsv[2]))
            file.write("HSV interval for 90% of masked points: [{:.0f}-{:.0f}, {:.0f}-{:.0f}, {:.0f}-{:.0f}]\n".format(low95hsv[0], high95hsv[0], low95hsv[1], high95hsv[1], low95hsv[2], high95hsv[2]))
        cv2.putText(res, "Data saved to {}".format(args.calibration_file), (100, res.shape[0] - 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.imshow('res', res)
        cv2.waitKey(1000)
    if k == 27:
        break

cv2.destroyAllWindows()
#cap.release()