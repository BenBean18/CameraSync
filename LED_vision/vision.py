import cv2, numpy, math, time
import numpy as np
from get_time import *
import cProfile
# set up video capture object
cap = cv2.VideoCapture("IMG_9081.MOV")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters_create()
# aruco_params.maxErroneousBitsInBorderRate = 1.0
aruco_params.useAruco3Detection = False
aruco_params.polygonalApproxAccuracyRate = 0.02
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.cornerRefinementMinAccuracy = 0.02 # orig 0.02
aruco_params.errorCorrectionRate = 1.0 # orig 1.0

def charuco(frame, ids = [0, 1]):
    # aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard_create(2, 2, 0.084, 0.066, aruco_dict)
    board.ids = ids
    # aruco_params = cv2.aruco.DetectorParameters_create()
    # aruco_params.maxErroneousBitsInBorderRate = 1.0
    # aruco_params.useAruco3Detection = False
    # aruco_params.polygonalApproxAccuracyRate = 0.05
    # aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # aruco_params.cornerRefinementMinAccuracy = 0.02
    # aruco_params.errorCorrectionRate = 1
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    avgPixelsPerAruco = 0
    avgAngle = 0
    # 0 1
    #  x
    # 3 2
    if len(corners) > 0:
        for corns in corners:
            # print(corns)
            avgPixelsPerAruco += abs((corns[0][1]-corns[0][0])[0]) + abs((corns[0][2]-corns[0][3])[0])
            avgPixelsPerAruco += abs((corns[0][3]-corns[0][0])[1]) + abs((corns[0][2]-corns[0][1])[1])
            avgAngle += math.atan2((corns[0][1]-corns[0][0])[1], (corns[0][1]-corns[0][0])[0])
        avgPixelsPerAruco /= len(corners) * 4
        avgAngle /= len(corners)
        pixelsPerMeter = avgPixelsPerAruco / 0.066
        return (cv2.aruco.interpolateCornersCharuco(corners, ids, frame, board), pixelsPerMeter, avgAngle)
    else:
        return ((0, corners, ids), 0, 0)

def aruco(frame):
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    return (corners, ids)

# https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

# need to somehow combine aruco detection in deskew and charuco...

lastAruco = (None, None)

def deskew(i: cv2.Mat):
    global lastAruco
    # only run detection every other frame
    if lastAruco == (None, None):
        (corners, ids) = aruco(i)
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
        # print("Couldn't deskew:", e)
        pass
    # corn = corners[2]
    # for line in [[1,0],[2,1],[2,3],[3,0]]:
    #     (xDif, yDif) = corn[0][line[0]] - corn[0][line[1]]
    #     avgHypot += math.hypot(xDif, yDif)
    # avgHypot /= len(corn[0])
    # print(avgHypot)
    # cornTL = corn[0][0]
    # cornTR = cornTL + (avgHypot, 0)
    # cornBR = cornTL + (avgHypot, avgHypot)
    # cornBL = cornTL + (0, avgHypot)
    # newCorn = np.float32([cornTL, cornTR, cornBR, cornBL])
    # print(corn[0], newCorn)
    # ptf = cv2.getPerspectiveTransform(corn[0], newCorn)
    # i2 = cv2.warpPerspective(i, ptf, i.shape[:2], flags=cv2.INTER_LINEAR)

    # need to do:
    # id2 TL
    # id0 TL
    # id1 BR
    # id3 BR

    # dest:
    # 1 pixel per millimeter
    # (0, 150)
    # (970, 165)
    # (1050, 315)
    # (150, 280)

    # 965 x 330

#32,46 -> 1300,120

def processFrame(frame: cv2.Mat):
    start = time.time()
    orig = frame
    # cv2.imshow("Original", frame)
    # cv2.waitKey(1)
    deskewed = deskew(frame)
    if not deskewed is None:
        frame = deskewed
    # cv2.imshow("Deskewed", frame)
    # cv2.waitKey(1)
    if deskewed is None:
        # deskewing does rotation so only spend CPU time doing this if it's None
        ((num01, corners01, ids01), ppm01, angle01) = charuco(frame, ids=[0, 1]) # right
        ((num23, corners23, ids23), ppm23, angle23) = charuco(frame, ids=[2, 3]) # left

        if (corners01 is None or len(corners01) == 0) and (corners23 is None or len(corners23) == 0):
            angle = 0
        elif (corners01 is None or len(corners01) == 0):
            angle = angle23
        elif (corners23 is None or len(corners23) == 0):
            angle = angle01
        else:
            angle = angle23 * 0.1 + angle01 * 0.1 + math.atan2(corners01[0][0][1] - corners23[0][0][1], corners01[0][0][0] - corners23[0][0][0]) * 0.8

        aAngle = math.degrees(angle)

        frame = rotate_image(frame, aAngle)

        ((num01, corners01, ids01), ppm01, angle01) = charuco(frame, ids=[0, 1]) # right
        ((num23, corners23, ids23), ppm23, angle23) = charuco(frame, ids=[2, 3]) # left

        # frame = cv2.aruco.drawDetectedCornersCharuco(frame, corners01, ids01)
        # frame = cv2.aruco.drawDetectedCornersCharuco(frame, corners23, ids23)

        mToTop = -0.20
        mToBottom = 0.07
        xOffset = -0.02
        xLength = 0.98
        # Should be able to put all of this within deskewing code -- just crop image properly
        if (corners01 is None or len(corners01) == 0) and (corners23 is None or len(corners23) == 0):
            # print(":( no ArUco markers found")
            return (None, None, None)
        elif (corners01 is None or len(corners01) == 0):
            angle = angle23
            if deskewed is None:
                pixelsPerMeter = ppm23
            else:
                pixelsPerMeter = 1000
            minX = int(corners23[0][0][0] - pixelsPerMeter * 0.06 + pixelsPerMeter * xOffset)
            minX = minX if minX >= 0 else 0
            # 8 cm to the top of the checkerboard and about 10 cm more to the top of the LED strip
            minY = int(corners23[0][0][1] + pixelsPerMeter * mToTop)
            minY = minY if minY >= 0 else 0
            maxY = int(minY + pixelsPerMeter * mToBottom) # exclude checkerboard
            maxX = int(minX + pixelsPerMeter * xLength)
        elif (corners23 is None or len(corners23) == 0):
            angle = angle01
            if deskewed is None:
                pixelsPerMeter = ppm01
            else:
                pixelsPerMeter = 1000
            minX = int(corners01[0][0][0] - pixelsPerMeter + pixelsPerMeter * xOffset)
            minX = minX if minX >= 0 else 0
            # 8 cm to the top of the checkerboard and about 10 cm more to the top of the LED strip
            minY = int(corners01[0][0][1] + pixelsPerMeter * mToTop)
            minY = minY if minY >= 0 else 0
            maxY = int(minY + pixelsPerMeter * mToBottom) # exclude checkerboard
            maxX = int(minX + pixelsPerMeter * xLength)
        else:
            if deskewed is None:
                pixelsPerMeter = (ppm01 + ppm23)/2 # need to find a way to deskew based on this difference
            else:
                pixelsPerMeter = 1000 # ... how did I forget this is true
            # print(ppm01, ppm23, pixelsPerMeter)
            # print(pixelsPerMeter)
            # strip is 1m long
            minX = int(corners23[0][0][0] - pixelsPerMeter * 0.06 + pixelsPerMeter * xOffset)
            minX = minX if minX >= 0 else 0

            maxX = int(corners01[0][0][0] + (pixelsPerMeter * (xLength - 1.0)))
            maxX = maxX if maxX <= frame.shape[1] else frame.shape[1]
            # 8 cm to the top of the checkerboard and about 10 cm more to the top of the LED strip
            minY = int((corners01[0][0][1] + pixelsPerMeter * mToTop + corners23[0][0][1] + pixelsPerMeter * mToTop) / 2)
            minY = minY if minY >= 0 else 0
            maxY = int(minY + pixelsPerMeter * mToBottom) # exclude checkerboard
        # print(minY, maxY, minX, maxX)
        # frame = cv2.circle(frame, (int(minX), int(minY)), 10, (255, 255, 255), -1)
        # frame = cv2.circle(frame, (int(maxX), int(maxY)), 10, (255, 255, 255), -1)
        frame = frame[minY:maxY, minX:maxX]
    else:
        # frame = frame[36:96, 50:1000] # if it's deskewed this should be constant and saves CPU time doing costly ArUco detection
        frame = frame[36:96, 40:1010] # if it's deskewed this should be constant and saves CPU time doing costly ArUco detection
    try:
        strip0 = getStrip(frame, 0)
        idx = 0
        zeroIndices = []
        for led in strip0:
            if led > 1 and led < 59:
                # frame = cv2.circle(frame, (int(idx/60 * (maxX - minX)), int(10)), 10, (255, 255, 255), -1)
                zeroIndices.append(led)
            idx += 1
        oneIndices = []
        strip1 = getStrip(frame, 1)
        idx = 0
        for led in strip1:
            if led > 1 and led < 59:
                # frame = cv2.circle(frame, (int(idx/60 * (maxX - minX)), int(((maxY - minY) - 10))), 10, (255, 255, 255), -1)
                oneIndices.append(led)
            idx += 1
        return (zeroIndices, oneIndices, frame)
    except Exception as e:
        print("Sadness", e)
        return (None, None, None)

#doStuff()
#cProfile.run("doStuff()")

def mainloop(cap: cv2.VideoCapture, debug: bool = False):
    # use readAllAtOnce for videos & disable for live camera
    lastTimestamp = 0
    deltas = []
    ts3 = 0
    ts25 = 0
    last3 = 0
    last25 = 0
    dropped = 0
    frames = 0
    stop = False
    epsilon = 0.0001
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
                    frame = cv2.circle(frame, ((int(i/60 * frame.shape[1])), int(10)), 10, (255, 255, 255), -1)
                for i in oneIndices:
                    frame = cv2.circle(frame, ((int(i/60 * frame.shape[1])), int(frame.shape[0] - 10)), 10, (255, 255, 255), -1)
            new3 = zeroIndices[0]
            new25 = oneIndices[0]
            ts3 += (new3 - last3) * (3/1000)
            ts25 += (new25 - last25) * (25/1000)
            last3 = new3
            last25 = new25
            ts = ts3 # relying on just the 3ms timestamp seems to work ok
            delta = (ts - lastTimestamp) % (3*58/1000)
            if delta > 0:
                if debug:
                    deltas.append(delta)
                    if len(deltas) > 100:
                        deltas = deltas[1:]
                    sortedDeltas = np.sort(deltas)
                    print("Speed:", f"{1/(time.time() - start):.4f}fps", "Timestamp:", f"{ts:.4f}", "Δ:", f"{delta:.4f}", "Mean Δ (25%-75%):", f"{np.mean(sortedDeltas[int(sortedDeltas.size*0.25):int(sortedDeltas.size*0.75)]):.4f}", "Diff:", f"{(time.time() - start) - delta:.4f}", "Shutter:", f"{len(zeroIndices)*(3/1000):.4f}ms", end="\r")
                else:
                    print("Speed:", f"{1/(time.time() - start):.4f}fps", "Timestamp:", f"{ts:.4f}", "Δ:", f"{delta:.4f}", "Diff:", f"{(time.time() - start) - delta:.4f}", "Shutter:", f"{len(zeroIndices)*(3/1000):.4f}ms", end="\r")
                # if delta is too high and it's not right at the end (red + green is harder to detect so might be seen as dropped)
                if delta > math.ceil(1/60 * (1000/3)) * (3/1000) + epsilon and not 59 in zeroIndices and not 58 in zeroIndices and not 57 in zeroIndices and not 0 in zeroIndices and not 1 in zeroIndices and not 2 in zeroIndices:
                    if debug:
                        print("\n****DROPPED****", frames, "Speed:", f"{1/(time.time() - start):.4f}fps", "Timestamp:", f"{ts:.4f}", "Δ:", f"{delta:.4f}", "Mean Δ (25%-75%):", f"{np.mean(sortedDeltas[int(sortedDeltas.size*0.25):int(sortedDeltas.size*0.75)]):.4f}", "Diff:", f"{(time.time() - start) - delta:.4f}", "Shutter:", f"{len(zeroIndices)*(3/1000):.4f}ms")
                    else:
                        print("\n****DROPPED****", frames, "Speed:", f"{1/(time.time() - start):.4f}fps", "Timestamp:", f"{ts:.4f}", "Δ:", f"{delta:.4f}", "Diff:", f"{(time.time() - start) - delta:.4f}", "Shutter:", f"{len(zeroIndices)*(3/1000):.4f}ms")
                    dropped += 1
                    k = cv2.waitKey(0)
                    if k == ord('q'):
                        return
                    if k == ord(' '):
                        cv2.waitKey(0)
                    if k == ord('n'):
                        stop = True
            else:
                lastTimestamp = 0
                print("Timestamp: ", round(ts, 4), "                                    ", end="\r")
                pass
            lastTimestamp = ts
            if debug:
                cv2.imshow("LEDs", frame)
                if not stop:
                    k = cv2.waitKey(1)
                else:
                    k = cv2.waitKey(0)
                if k == ord('q'):
                    return
                if k == ord(' '):
                    cv2.waitKey(0)
                if k == ord('n'):
                    stop = True
        except Exception as e:
            print(e, "Couldn't find LEDs")
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