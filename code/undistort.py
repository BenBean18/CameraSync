import cv2, sys, time
import numpy as np

calibration_data = {"mtx": [[740.000487833163, 0.0, 713.8475781511161], [0.0, 686.99438662146, 712.6682577725227], [0.0, 0.0, 1.0]], "dist": [[-0.1073197789401041, -0.023463643170938138, -0.005596367039274652, -0.007053906685202043, 0.004180178882056824]], "cam": "aria"}
mtx = np.array(calibration_data["mtx"], dtype=np.float32)
dist = np.array(calibration_data["dist"], dtype=np.float32)

f = sys.argv[1].split(".")

cap = cv2.VideoCapture(f[0] + "." + f[1])
writer = cv2.VideoWriter(f[0] + "_undistorted." + f[1], cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

i = 0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    dst = cv2.undistort(frame, mtx, dist, None, mtx)
    writer.write(dst)
    print(f"{str(i).zfill(len(str(total)))}/{total} [{round(i * 100/total, 1)}%] {round(i/(time.time() - start))}fps", end="\r")
    i += 1