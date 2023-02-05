# import libraries
import cv2
import socket
import sys
import time
import threading

# connect to LED strip for configuration
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s.connect(("ledStrip.local", 12345))

s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s2.connect(("ledStrip2.local", 12345))

# functions to configure LED strip (delay time and gap between on LEDs)
def setDelay(delay, strip=1):
    global s, s2
    s_ = s if strip == 1 else s2
    #s_.sendall(bytes("d"+str(delay)+"\n\n", "utf-8"))

def setGap(gap, strip=1):
    global s, s2
    s_ = s if strip == 1 else s2
    #s_.sendall(bytes("g"+str(gap)+"\n\n", "utf-8"))

# read configuration values from command line arguments
try:
    d1 = int(sys.argv[1])
    g1 = int(sys.argv[2])
    d2 = int(sys.argv[3])
    g2 = int(sys.argv[4])
    t = int(sys.argv[5]) # time in seconds
except:
    print("Usage: ")
    print("camera_sync_code.py <s1 delay> <s1 gap> <s2 delay> <s2 gap> <how long to record for>")
    print("Suggested configuration: ")
    print("camera_sync_code.py 3 0 25 0 <time>")

# send configuration to the LED strip
setDelay(d1)
setGap(g1)
setDelay(d2, strip=2)
setGap(g2, strip=2)

# function to record a video
def captureVideo(cap):
    # set up video capture object
    vid = cv2.VideoCapture(cap, cv2.CAP_MSMF)

    # record the start time
    start = time.time()

    # create a list of frames (to be populated later)
    bigframes = []

    # set to 60fps ideally, the cameras I was using did ~25
    vid.set(cv2.CAP_PROP_FPS, 30)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # print out that this video capture has started
    print(f"Start {cap} @ {time.time()}")

    # read one "throwaway" frame to initialize camera
    # I added this after the video appeared to go backwards then forwards in the first 3 frames
    
    # changed it to more frames so cameras adjust white balance
    # and this brings both cameras up to 30 fps?!?!?! yay!!!
    while (time.time()-start) < 5:
        _, frame = vid.read()
    
    start = time.time()

    print(f"Actually recording {cap} @ {start} ~== {time.time()}")

    # capture video for         t seconds
    while (time.time()-start) < t:
        # read video frame
        _, frame = vid.read()

        # append frame to list of frames.
        # not writing it to the video file right now for optimization
        # but I'm not sure if this actually optimizes anything
        bigframes.append(frame)

    # calculate recorded FPS (# frames recorded / t seconds)
    print(len(bigframes))
    fps = len(bigframes)/float(t)
    # set output video format to be MP4
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # create video output object
    # filename format: video_sync_<fps>fps_<ms each LED is on>ms_gap<gap between LEDs>_cam<camera ID>.mp4
    output = cv2.VideoWriter('video_sync_'+str(fps)+'fps_cam'+str(cap)+'.mp4', fourcc, fps, (1280,720))

    # write recorded frames to video file
    for f in range(len(bigframes)):
        output.write(bigframes[f])

    # release camera object
    vid.release()

# create two threads to record video, one for each camera
#threading.Thread(target=lambda: captureVideo(2)).start()
#threading.Thread(target=lambda: captureVideo(1)).start()
captureVideo(1)