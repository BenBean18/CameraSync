import cv2, numpy, math, time
import numpy as np
from get_time import *
import cProfile
# set up video capture object
cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture("./LED_vision/IMG_9082.MOV")

class TagDetector:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()
    # aruco_params.maxErroneousBitsInBorderRate = 1.0
    aruco_params.useAruco3Detection = False
    aruco_params.polygonalApproxAccuracyRate = 0.02
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_params.cornerRefinementMinAccuracy = 0.02 # orig 0.02
    aruco_params.errorCorrectionRate = 1.0 # orig 1.0

    def pixelsPerMeterAndAngle(corners):
        """
        Given the corners of a ChArUco board, return the angle of the board 
        and the number of pixels per meter (assuming the tags are 6.6cm across)

        * `corners`: the corners of the board
        """
        avgPixelsPerAruco = 0
        avgAngle = 0
        for corns in corners:
            avgPixelsPerAruco += abs((corns[0][1]-corns[0][0])[0]) + abs((corns[0][2]-corns[0][3])[0])
            avgPixelsPerAruco += abs((corns[0][3]-corns[0][0])[1]) + abs((corns[0][2]-corns[0][1])[1])
            avgAngle += math.atan2((corns[0][1]-corns[0][0])[1], (corns[0][1]-corns[0][0])[0])
        avgPixelsPerAruco /= len(corners) * 4
        avgAngle /= len(corners)
        return (avgPixelsPerAruco / 0.066, avgAngle)

    def charuco(frame, ids = [0, 1]):
        """
        Detect a ChArUco board.

        2x2, where each tag is 6.6cm across & each square is 8.4cm across.

        Returns ((number of tags, corners, ids), pixels per meter, angle of tags)

        * `frame`: the input image
        * `ids`: the IDs of the ArUco markers
        """
        board = cv2.aruco.CharucoBoard_create(2, 2, 0.084, 0.066, TagDetector.aruco_dict)
        board.ids = ids
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, TagDetector.aruco_dict, parameters=TagDetector.aruco_params)
        avgPixelsPerAruco = 0
        avgAngle = 0
        # corner ids:
        # 0 1
        #  x
        # 3 2
        if len(corners) > 0:
            pixelsPerMeter, avgAngle = TagDetector.pixelsPerMeterAndAngle(corners)
            return (cv2.aruco.interpolateCornersCharuco(corners, ids, frame, board), pixelsPerMeter, avgAngle)
        else:
            return ((0, corners, ids), 0, 0)

    def aruco(frame):
        """
        Detects ArUco markers.

        * `frame`: the input image
        """
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, TagDetector.aruco_dict, parameters=TagDetector.aruco_params)
        return (corners, ids)

lastAruco = (None, None)

def deskew(i: cv2.Mat):
    """
    Return a deskewed, cropped version of `i`.

    * `i`: the input image
    """
    global lastAruco
    # only run detection every other frame
    if lastAruco == (None, None):
        (corners, ids) = TagDetector.aruco(i)
        lastAruco = (corners, ids)
    else:
        (corners, ids) = lastAruco
        lastAruco = (None, None)
    # 0 1
    #  x
    # 3 2
    # find average hypotenuse
    # lines: 1 - 0
    #        2 - 1
    #        2 - 3
    #        3 - 0
    avgHypot = 0
    try:
        zero = corners[np.where(ids == [0])[0][0]]
        one = corners[np.where(ids == [1])[0][0]]
        two = corners[np.where(ids == [2])[0][0]]
        three = corners[np.where(ids == [3])[0][0]]
        tl2 = two[0][0]
        tl0 = zero[0][0]
        br1 = one[0][2]
        br3 = three[0][2]
        oldRect = np.float32([tl2, tl0, br1, br3])
        # id2 TL
        # id0 TL
        # id1 BR
        # id3 BR
        newRect = np.float32([[50, 150], [950, 165], [1100, 315], [200, 300]])
        ptf = cv2.getPerspectiveTransform(oldRect, newRect)
        i2 = cv2.warpPerspective(i, ptf, (1200, 350), flags=cv2.INTER_LINEAR)
        return i2
    except Exception as e:
        pass

def clamp(val, a, b):
    if val < a:
        return a
    if val > b:
        return b
    return val

def processFrame(frame: cv2.Mat, drawLEDLines: bool = False):
    """
    Processes a frame and returns the positions of LEDs in both strips.

    Returns (`strip0Indices`, `strip1Indices`, `deskewed_cropped_image`)

    * `frame`: input image
    * `drawLEDLines`: if true, draws vertical lines where LEDs are on the image
    """
    deskewed = deskew(frame)
    if not deskewed is None:
        frame = deskewed
    
    xOffset = 0.05 # must be +
    xLength = 0.97
    if deskewed is None:
        return (None, None, None)
    else:
        # Crop deskewed image
        h, w, *_ = frame.shape
        frame = frame[36:96, clamp(int((xOffset*1000)), 0, w):clamp(int(xOffset*1000+xLength*1000), 0, w)] # if it's deskewed this should be constant and saves CPU time doing costly ArUco detection
    try:
        # Get LEDs on in strip 0
        strip0 = getStrip(frame, 0)
        idx = 0
        zeroIndices = []
        for led in strip0:
            if led > 1 and led < 59:
                # frame = cv2.circle(frame, (int(idx/60 * (maxX - minX)), int(10)), 10, (255, 255, 255), -1)
                zeroIndices.append(led)
            idx += 1
        # Get LEDs on in strip 1
        oneIndices = []
        strip1 = getStrip(frame, 1)
        idx = 0
        for led in strip1:
            if led > 1 and led < 59:
                # frame = cv2.circle(frame, (int(idx/60 * (maxX - minX)), int(((maxY - minY) - 10))), 10, (255, 255, 255), -1)
                oneIndices.append(led)
            idx += 1
        # Draw lines where LEDs are (if applicable)
        if drawLEDLines:
            h, w_, *_ = frame.shape
            w = getLEDWidth(frame)
            # w = 5
            g = getLEDGap(frame)
            # g = (w_ - (5 * 60)) / 59
            for ledNum in range(60):
                minPos = int((w + g) * ledNum)
                maxPos = int(minPos + w + g)
                frame = cv2.rectangle(frame, (minPos, 0), (maxPos, h), (255,255,255), 1)
        return (zeroIndices, oneIndices, frame)
    except Exception as e:
        print("Sadness", e)
        return (None, None, None)

#doStuff()
#cProfile.run("doStuff()")

lastTimestamp = 0
def mainloop_new(cap: cv2.VideoCapture):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            (zeroIndices, oneIndices, frame) = processFrame(frame)
            zero = zeroIndices[0]
            one = oneIndices[0]
            ts = timestamp(zero, one)
            print(ts, "                                                 ", end="\r")
        except Exception as e:
            print(e, "Couldn't find LEDs")
            lastFrameGood = False
            pass

def mainloop(cap: cv2.VideoCapture, debug: bool = False, drawLines: bool = False, bw: bool = False):
    # use readAllAtOnce for videos & disable for live camera
    lastTimestamp = 0
    dropped = 0
    frames = 0
    stop = False
    while True:
        frames += 1
        #print(frames)
        ret, frame = cap.read()
        if not ret:
            if debug:
                print("error with image")
            break
        start = time.time()
        try:
            (zeroIndices, oneIndices, frame) = processFrame(frame)
            if debug:
                for i in zeroIndices:
                    frame = cv2.circle(frame, ((int(i/60 * frame.shape[1])), int(15)), 15, (255, 255, 255), -1)
                for i in oneIndices:
                    frame = cv2.circle(frame, ((int(i/60 * frame.shape[1])), int(frame.shape[0] - 15)), 15, (255, 255, 255), -1)
            zero = zeroIndices[0]
            one = oneIndices[-1] # last index for this, we want the latest
            ts = timestamp(zero, one)
            dropped += 1 if lastTimestamp > ts else 0
            print(one, zero, ts, "*" if lastTimestamp > ts else "")
            lastTimestamp = ts
            if debug:
                if bw:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    frame_threshold = cv2.inRange(hsv, (0, 0, 250), (30, 255, 255)) # value min was 245 but this works better
                    frame_threshold = cv2.bitwise_or(frame_threshold, cv2.inRange(hsv, (170, 0, 250), (180, 255, 255))) # value min was 245 but this works better
                    eroded = cv2.erode(frame_threshold, None, iterations = 0)
                    h, w_, *_ = frame.shape
                    w = getLEDWidth(frame)
                    # w = 5
                    g = getLEDGap(frame)
                    # g = (w_ - (5 * 60)) / 59
                    for ledNum in range(60):
                        minPos = int((w + g) * ledNum)
                        maxPos = int(minPos + w + g)
                        eroded = cv2.rectangle(eroded, (minPos, 0), (maxPos, h), (255,255,255), 1)
                    cv2.imshow("LEDs", eroded)
                else:
                    cv2.imshow("LEDs", frame)
                if not stop:
                    k = cv2.waitKey(1)
                else:
                    k = cv2.waitKey(0)
                if k == ord('q'):
                    stop = False
                    break
                if k == ord(' '):
                    cv2.waitKey(0)
                    stop = False
                if k == ord('n'):
                    stop = True
            lastFrameGood = True
        except Exception as e:
            print(e, "Couldn't find LEDs")
            lastFrameGood = False
            pass
    print(f"{dropped}/{frames} dropped frames ({round(dropped / frames * 100)}%)")

import threading
lastTimestamp = 0
deltas = []
ts3 = 0
ts25 = 0
last3 = 0
last25 = 0
dropped = 0
frames = 0
threadCount = 0

tcM = threading.Lock()

def getTimestamp(frame: cv2.Mat, tc: bool = False):
    if tc:
        global threadCount
        tcM.acquire()
        threadCount += 1
        tcM.release()
    try:
        (zeroIndices, oneIndices, frame) = processFrame(frame)
        ts = timestamp(zeroIndices[0], 0)
        if tc:
            tcM.acquire()
            threadCount -= 1
            tcM.release()
        return ts
    except:
        if tc:
            tcM.acquire()
            threadCount -= 1
            tcM.release()
        return None

maxThreads = 4

def asyncMainloop(cap: cv2.VideoCapture, debug: bool = False):
    global threadCount, maxThreads, frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("error with image")
            break
        frames += 1
        if threadCount > maxThreads:
            print(frames, getTimestamp(frame, False))
        else:
            threading.Thread(target=lambda: print(frames, getTimestamp(frame, True))).start()

# async runs faster and gives timestamp for each image
start = time.time()
cProfile.run("mainloop(cap, True)", sort="cumtime") # sort by longest time first
print(time.time() - start)

cap.set(0, 1)

def getTimestampIfNotOnEnd(frame):
    try:
        (zeroIndices, oneIndices, frame) = processFrame(frame)
        # first need to find offset between 3 and 25 before using
        ts = timestamp(zeroIndices[0], 0)
        if not 59 in zeroIndices and not 58 in zeroIndices and not 57 in zeroIndices and not 0 in zeroIndices and not 1 in zeroIndices and not 2 in zeroIndices:
            return ts
        else:
            return None
    except:
        return None

# actual: 9081 is a bit ahead
def videoCompare(cap1: cv2.VideoCapture, cap2: cv2.VideoCapture):
    # Prints out (cap1 - cap2)
    last1 = 0
    last2 = 0
    while True:
        ret, frame1 = cap1.read()
        if not ret:
            print("error with image")
            break
        ret, frame2 = cap2.read()
        if not ret:
            print("error with image")
            break
        ts1 = getTimestampIfNotOnEnd(frame1)
        
        ts2 = getTimestampIfNotOnEnd(frame2)
        
        if ts1 != None and ts2 != None:
            # 58 b/c ignoring 0 and 59, see bottom of processFrame
            delta1 = (ts1 - last1) % (3*58/1000)
            t1 = last1 + delta1
            delta2 = (ts2 - last2) % (3*58/1000)
            t2 = last2 + delta2
            # if you do  % (3*58/1000) on the difference, then it will work better (e.g. if frame 1 drops out for a bit, messing up the absolute timestamp)
            # but ONLY IF video 1 is actually ahead
            print(f"#1: {t1:.4f}, #2: {t2:.4f}, #1 - #2: {(t1-t2):.4f}", end="\n")
            last1 = t1
            last2 = t2
        else:
            print("none")

#videoCompare(cap, cap2)