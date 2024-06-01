import cv2

alg = "haarcascade_frontalface_default.xml"

haar_cascade = cv2.CascadeClassifier(alg)

cam = cv2.VideoCapture(0)

while True:
    text = "No Person Detected"
    # Read camera feeds
    _,img = cam.read()
    # convert to gray scale
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    # perform haarcascade detection
    face = haar_cascade.detectMultiScale(grayImg,1.3,4)
    # draw rectangle on face
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # set text = No Person Detected
        text = "Person Detected"
    # put text on screen
    cv2.putText(img, text, (10,20),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
    # show the screen
    cv2.imshow("Person Detection",img)
    key = cv2.waitKey(10)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
