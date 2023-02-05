import cv2, sys
cap = cv2.VideoCapture(sys.argv[1])
cap.set(1, int(sys.argv[2]))
ret, frame = cap.read()
cv2.imwrite(f"ext_frame_{sys.argv[2]}_from_{sys.argv[1]}.jpg", frame)