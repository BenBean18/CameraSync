import cv2, math
import numpy as np

LED_COUNT = 60
STRIP_0_LED_TIME = 1 # ms
STRIP_1_LED_TIME = STRIP_0_LED_TIME * (LED_COUNT - 2) # ms

def frameTime(firstLED, lastLED):
    return (lastLED - firstLED) * (STRIP_0_LED_TIME/1000)

def timestamp(strip0: int, strip1: int):
    """
    Calculates the timestamp using the position of the LEDs in both strips.
    
    * `strip0`: the position of the first LED in the top strip
    * `strip1`: the position of the first LED in the bottom strip
    """
    return (strip0 * STRIP_0_LED_TIME/1000) + (strip1 * STRIP_1_LED_TIME/1000)

def green(img: cv2.Mat):
    """
    Filters an image for green pixels.

    * `img`: the input image
    """
    img = cv2.blur(img, (3,3)) # was blur (3,3)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # minsat was 50
    frame_threshold = cv2.inRange(hsv, (84, 0, 156), (101, 100, 255))
    dilated = cv2.dilate(frame_threshold, None, iterations=2) # was 2 iters
    return dilated

def getLEDWidth(img):
    """
    Gets the width of an individual LED.
    The image is cropped using the positions of ArUco markers so that 1px = 1mm.
    These are 5mm LEDs, so just return 5.
    The commented out code is for dynamically figuring out the LED width.
    """
    return 5
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
    # median:
    # return samples[(len(samples)-1)//2] if len(samples) % 2 != 0 else (samples[len(samples)//2]+samples[len(samples)//2-1])/2

def getLEDGap(img: cv2.Mat):
    """
    Computes the gap between LEDs.

    * `img`: the input image
    """
    width = getLEDWidth(img) * 60
    h, w, *_ = img.shape
    pixelsLeft = w - width
    gap = pixelsLeft / 59
    # 59 gaps between
    # ex. with 3 LEDs, 2 gaps
    # *.*.*
    return gap

def getLEDSliceOrig(img: cv2.Mat, ledNum: int, w, g):
    """
    Crop an image to get a specific LED (in both strips).
    Calling getLEDWidth and getLEDGap 60 times doesn't make sense, so we pass 
    in the outputs.

    * `img`: the input image
    * `ledNum`: the LED position
    * `w`: the output of getLEDWidth
    * `g`: the output of getLEDGap
    """
    minPos = int((w + g) * ledNum)
    maxPos = int(minPos + w + g)
    h, w, *_ = img.shape
    return img[0:h, minPos:maxPos]

def getLEDSlice(img: cv2.Mat, ledNum: int, stripNum: int, w, g):
    """
    Crop an image to get a specific LED (in one strip).
    Calling getLEDWidth and getLEDGap 60 times doesn't make sense, so we pass 
    in the outputs.

    * `img`: the input image
    * `ledNum`: the LED position
    * `stripNum`: the LED strip (0 = top, 1 = bottom)
    * `w`: the output of getLEDWidth
    * `g`: the output of getLEDGap
    """
    img = getLEDSliceOrig(img, ledNum, w, g)
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
    """
    Checks to see if a LED is on in the cropped HSV image.

    * `hsv`: the input image, in the HSV color space
    """
    frame_threshold = cv2.inRange(hsv, (13, 0, 250), (50, 150, 255)) # value min was 245 but this works better
    frame_threshold = cv2.bitwise_or(frame_threshold, cv2.inRange(hsv, (170, 0, 250), (180, 150, 255))) # value min was 245 but this works better
    eroded = cv2.erode(frame_threshold, None, iterations = 0)
    return cv2.countNonZero(eroded) > 0

# def getStrip(img: cv2.Mat, ledStrip: int):
#     w = getLEDWidth(img)
#     g = getLEDGap(img)
#     img = red(img)
#     on = [i if cv2.countNonZero(getLEDSlice(img, i, ledStrip, w, g)) > 0 else 0 for i in range(60)]
#     return on

def getStrip(img: cv2.Mat, ledStrip: int):
    """
    Returns all LEDs that are on in a strip.

    * `img`: the input image
    * `ledStrip`: the strip ID (0 = top, 1 = bottom)
    """
    w = getLEDWidth(img)
    g = getLEDGap(img)
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    on = [i+1 if isOn(getLEDSlice(h, i, ledStrip, w, g)) else 0 for i in range(59)]
    return on

def x(contour):
    """
    Gets the X position of an OpenCV contour.

    * `contour`: the contour
    """
    x,y,w,h = cv2.boundingRect(contour)
    return x

def y(contour):
    """
    Gets the Y position of an OpenCV contour.

    * `contour`: the contour
    """
    x,y,w,h = cv2.boundingRect(contour)
    return y

# https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
def rotate_image(image, angle):
    """
    Rotates `image` by `angle` degrees.

    * `img`: the input image
    * `angle`: the angle (in degrees) to rotate it
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def getAngle(img: cv2.Mat):
    """
    Gets the angle to rotate `img` to based on the positions of the green LEDs.
    For example, if the green LEDs look like this:
    ##
     ..
      ..
       ##
    then the angle will be approximately -45 degrees (I think).

    * `img`: the input image
    """
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
    """
    Rotates `img` to the correct position based on the positions of the green 
    LEDs.

    * `img`: the input image
    """
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