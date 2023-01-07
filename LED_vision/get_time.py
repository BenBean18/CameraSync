import cv2, math
import numpy as np

def frameTime(firstLED, lastLED):
    return (lastLED - firstLED) * (3/1000)

def timestamp(strip0, strip1):
    return (strip0 * 3/1000) + (strip1 * 25/1000)

def green(img):
    img = cv2.blur(img, (3,3)) # was blur (3,3)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # minsat was 50
    frame_threshold = cv2.inRange(hsv, (84, 0, 156), (101, 100, 255))
    dilated = cv2.dilate(frame_threshold, None, iterations=2) # was 2 iters
    return dilated

def getLEDWidth(img):
    # img2 = img
    # img = green(img)
    # # cv2.imshow("a", img)
    # # cv2.waitKey(0)
    # contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # samples = []
    # for contour in contours:
    #     x,y,w,h = cv2.boundingRect(contour)
    #     # (x,y),radius = cv2.minEnclosingCircle(contour)
    #     # center = (int(x),int(y))
    #     # radius = int(radius)
    #     # img2 = cv2.circle(img2,center,radius,(255,255,0),2)
    #     samples.append(w)
    # # cv2.imshow("a", img2)
    # # cv2.waitKey(0)
    # if len(samples) == 0:
    #     return 5 # with deskewing, 1px should equal 1mm and these are 5mm LEDs
    # else:
    #     return max(samples)
    return 5
    # median:
    # return samples[(len(samples)-1)//2] if len(samples) % 2 != 0 else (samples[len(samples)//2]+samples[len(samples)//2-1])/2

def getLEDGap(img: cv2.Mat):
    width = getLEDWidth(img) * 60
    h, w, *_ = img.shape
    pixelsLeft = w - width
    gap = pixelsLeft / 59
    # 59 gaps between
    # ex. with 3 LEDs, 2 gaps
    # *.*.*
    return gap

def getLEDSliceOrig(img: cv2.Mat, ledNum: int, w, g):
    minPos = int((w + g) * ledNum)
    maxPos = int(minPos + w + g)
    h, w, *_ = img.shape
    return img[0:h, minPos:maxPos]

def getLEDSlice(img_: cv2.Mat, ledNum: int, stripNum: int, w, g):
    img = getLEDSliceOrig(img_, ledNum, w, g)
    h, w, *_ = img.shape
    return img[(0 if stripNum == 0 else int(h/2)):(int(h/2) if stripNum == 0 else h), 0:w]

# Choose your own adventure:

# 1. Looks for all colors. EXPECTS HSV
# def isOn(hsv: cv2.Mat):
#     frame_threshold = cv2.inRange(hsv, (0, 0, 250), (180, 154, 255)) # value min was 245 but this works better
#     eroded = cv2.erode(frame_threshold, None, iterations = 2) # Changing it to no erosion (iterations = 0) REALLY helped
#     return cv2.countNonZero(eroded) > 0

# 2. Just looks for red. EXPECTS HSV. Doesn't work as well -- false positive dropped frames
def isOn(hsv: cv2.Mat):
    frame_threshold = cv2.inRange(hsv, (0, 0, 250), (30, 255, 255)) # value min was 245 but this works better
    frame_threshold = cv2.bitwise_or(frame_threshold, cv2.inRange(hsv, (170, 0, 250), (180, 255, 255))) # value min was 245 but this works better
    eroded = cv2.erode(frame_threshold, None, iterations = 0)
    return cv2.countNonZero(eroded) > 0

# def getStrip(img: cv2.Mat, ledStrip: int):
#     w = getLEDWidth(img)
#     g = getLEDGap(img)
#     img = red(img)
#     on = [i if cv2.countNonZero(getLEDSlice(img, i, ledStrip, w, g)) > 0 else 0 for i in range(60)]
#     return on

def getStrip(img: cv2.Mat, ledStrip: int):
    w = getLEDWidth(img)
    g = getLEDGap(img)
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    on = [i+1 if isOn(getLEDSlice(h, i, ledStrip, w, g)) else 0 for i in range(59)]
    return on

def x(contour):
    x,y,w,h = cv2.boundingRect(contour)
    return x

def y(contour):
    x,y,w,h = cv2.boundingRect(contour)
    return y

# https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def getAngle(img: cv2.Mat):
    g = green(img)
    h, w, *_ = img.shape
    contours, hierarchy = cv2.findContours(g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sortedContours = sorted(contours, key=lambda contour: x(contour))
    xa1,ya1,_,_ = cv2.boundingRect(sortedContours[0])
    xa2,ya2,_,_ = cv2.boundingRect(sortedContours[1])
    xb1,yb1,_,_ = cv2.boundingRect(sortedContours[-1])
    xb2,yb2,_,_ = cv2.boundingRect(sortedContours[-2])
    x1 = (xa1+xa2)/2
    y1 = (ya1+ya2)/2
    x2 = (xb1+xb2)/2
    y2 = (yb1+yb2)/2
    angle = math.atan2((y2-y1),(x2-x1))
    return math.degrees(angle)

def doRotation(img: cv2.Mat):
    angle = getAngle(img)
    return rotate_image(img, angle)

# im = cv2.imread("frame2_cropped.PNG")
# print(getLEDWidth(im))
# im = doRotation(im)
# cv2.imshow("whydoikeepforgettingtoputanamehere", green(im))
# cv2.waitKey(0)
# print(getLEDWidth(im))
# print(getLEDGap(im))
# print(getStrip(im, 0))
# print(getStrip(im, 1))

# Can use ArUco markers with known sizes to *know* the LED width
# We can also use them to calculate the angle
# Still need to crop to LED strip, probably using ArUco