import cv2, sys

background = cv2.imread(sys.argv[1]) # bg
overlay = cv2.imread(sys.argv[2]) # fg

added_image = cv2.addWeighted(background,0.7,overlay,0.7,0)

cv2.imwrite(f'{sys.argv[2]}_overlay.png', added_image)