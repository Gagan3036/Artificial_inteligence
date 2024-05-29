import cv2
img = cv2.imread("sample1.png")
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("GrayImg.jpg",grayImg)
cv2.imshow("GrayImg",grayImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
