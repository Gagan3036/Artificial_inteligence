import cv2
img = cv2.imread("sample2.png")
cv2.imshow("Pantech Logo",img)
cv2.imwrite("PantechLogo.png",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
