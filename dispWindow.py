from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import sys
import os

windowName = 'Drawing Demo'
img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow(windowName)

# true if mouse is pressed
drawing = False

# if True, draw rectangle. Press 'm' to toggle to curve
mode = False 
(ix, iy) = (-1, -1)

# mouse callback function
def draw_shape(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        (ix, iy) = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

cv2.setMouseCallback(windowName, draw_shape)

def main():
    global mode
    
    if len(sys.argv) != 2:
    	print('The run formation is python dispWindow.py filename')
    	print('filename: name of the file to save(do not enter the full path)')
    	print('Eg: python dispWindow.py img1.png')
    	sys.exit()
    else:
    	saved_filename = sys.argv[1]

    while(True):
        cv2.imshow(windowName, img)
        
        k = cv2.waitKey(1)
        if k == ord('m') or k == ord('M'):
            mode = not mode
        elif k == 27:		# Esc button
                cv2.imwrite('dataset/test_images/'+saved_filename,img)
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()