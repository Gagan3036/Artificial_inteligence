import cv2
import time

cam = cv2.VideoCapture(0)
time.sleep(0.5)

while True:
    _,img = cam.read()

    cv2.imshow("CameraFeed",img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
