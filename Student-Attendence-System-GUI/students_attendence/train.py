from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps, ImageDraw
from tkinter import messagebox
import mysql.connector
import cv2
import os
import numpy as np


class train:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1920x1200+0+0")
        self.root.title("Train data")

        img = Image.open(r"C:\\Users\\gagan\\Desktop\\FaceRecognition\\images\\face.jpg")
        self.photoimg = ImageTk.PhotoImage(img)

        bg_img = Label(self.root, image=self.photoimg)
        bg_img.place(x=0, y=0, width=1920, relheight=1)

        b1 = Button(self.root, text="Train Data", command=self.train_classifier, cursor="hand2",
                      font=("Helvetica", 30, "bold"), fg="white", bg="#42aad3", pady=10)
        b1.place(x=500, y=300, width=250, height=70)


    def train_classifier(self):
        data_dir = ("data")
        path = [os.path.join(data_dir,file) for file in os.listdir(data_dir)]

        faces = []
        ids = []

        for image in path:
            img = Image.open(image).convert('L')   # Gray scale image
            imageNp = np.array(img,'uint8')
            id = int(os.path.split(image)[1].split('.')[1])

            faces.append(imageNp)
            ids.append(id)
            cv2.imshow("Training",imageNp)
            cv2.waitKey(1) == 13
        ids = np.array(ids)


# ===========================Train the classifier and save==========================
        try:
            clf = cv2.face.LBPHFaceRecognizer_create()
            clf.train(faces,ids)
            clf.write("classifier.xml")
            cv2.destroyAllWindows()
            messagebox.showinfo("Result","Training datasets completed")
        except Exception as es:
                messagebox.showerror("Error", f"Due To: {str(es)}", parent=self.root)


if __name__ == "__main__":
    root = Tk()
    obj = train(root)
    root.mainloop()
