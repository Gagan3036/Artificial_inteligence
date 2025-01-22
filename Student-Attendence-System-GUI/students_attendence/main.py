from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps, ImageDraw
from student import Student
import os

class FaceRecognitionSystem:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1920x1200+0+0")
        self.root.title("Face Recognition System")

        # Load the main background image without resizing
        img = Image.open(r"C:\\Users\\gagan\\Desktop\\FaceRecognition\\images\\face.jpg")
        self.photoimg = ImageTk.PhotoImage(img)

        bg_img = Label(self.root, image=self.photoimg)
        bg_img.place(x=0, y=0, width=1920, relheight=1)

        # Add a title label with the custom font and background color
        title_lbl = Label(bg_img, text="FACE RECOGNITION ATTENDANCE SYSTEM", font=("Helvetica", 20, "bold"), fg="white", bg="#1b1b33", pady=10)
        title_lbl.place(x=85, y=150, anchor="w")

        # Student button
        img2 = Image.open(r"C:\\Users\\gagan\\Desktop\\FaceRecognition\\images\\student.png")
        img2 = img2.resize((100, 100), Image.LANCZOS)
        self.photoimg2 = ImageTk.PhotoImage(img2)

        b1 = Button(bg_img, image=self.photoimg2,command=self.student_details,cursor="hand2")
        b1.place(x=75, y=200, width=100, height=100)

        b1_1 = Button(bg_img, text="Students Details",command=self.student_details, cursor="hand2", font=("Helvetica",9, "bold"), fg="white", bg="#42aad3", pady=10)
        b1_1.place(x=75, y=300, width=100, height=30)

        # Detect Face button
        img3 = Image.open(r"C:\\Users\\gagan\\Desktop\\FaceRecognition\\images\\student.png")
        img3 = img3.resize((100, 100), Image.LANCZOS)
        self.photoimg3 = ImageTk.PhotoImage(img3)

        b1 = Button(bg_img, image=self.photoimg3, cursor="hand2")
        b1.place(x=250, y=200, width=100, height=100)

        b1_1 = Button(bg_img, text="Face Detector", cursor="hand2", font=("Helvetica", 9, "bold"), fg="white",
                      bg="#42aad3", pady=10)
        b1_1.place(x=250, y=300, width=100, height=30)

        # Attendance Face button
        img4 = Image.open(r"C:\\Users\\gagan\\Desktop\\FaceRecognition\\images\\student.png")
        img4 = img4.resize((100, 100), Image.LANCZOS)
        self.photoimg4 = ImageTk.PhotoImage(img4)

        b1 = Button(bg_img, image=self.photoimg4, cursor="hand2")
        b1.place(x=425, y=200, width=100, height=100)

        b1_1 = Button(bg_img, text="Attendance", cursor="hand2", font=("Helvetica", 9, "bold"), fg="white",
                      bg="#42aad3", pady=10)
        b1_1.place(x=425, y=300, width=100, height=30)

        # Help Face button
        img5 = Image.open(r"C:\\Users\\gagan\\Desktop\\FaceRecognition\\images\\student.png")
        img5 = img5.resize((100, 100), Image.LANCZOS)
        self.photoimg5 = ImageTk.PhotoImage(img5)

        b1 = Button(bg_img, image=self.photoimg5, cursor="hand2")
        b1.place(x=600, y=200, width=100, height=100)

        b1_1 = Button(bg_img, text="Help Desk", cursor="hand2", font=("Helvetica", 9, "bold"), fg="white",
                      bg="#42aad3", pady=10)
        b1_1.place(x=600, y=300, width=100, height=30)

        # Train Data button
        img6 = Image.open(r"C:\\Users\\gagan\\Desktop\\FaceRecognition\\images\\student.png")
        img6 = img6.resize((100, 100), Image.LANCZOS)
        self.photoimg6 = ImageTk.PhotoImage(img6)

        b1 = Button(bg_img, image=self.photoimg6, cursor="hand2")
        b1.place(x=75, y=400, width=100, height=100)

        b1_1 = Button(bg_img, text="Train Data", cursor="hand2", font=("Helvetica", 9, "bold"), fg="white",
                      bg="#42aad3", pady=10)
        b1_1.place(x=75, y=500, width=100, height=30)

        # Photo Data button
        img7 = Image.open(r"C:\\Users\\gagan\\Desktop\\FaceRecognition\\images\\student.png")
        img7 = img7.resize((100, 100), Image.LANCZOS)
        self.photoimg7 = ImageTk.PhotoImage(img7)

        b1 = Button(bg_img, image=self.photoimg7, cursor="hand2", command=self.open_img)
        b1.place(x=250, y=400, width=100, height=100)

        b1_1 = Button(bg_img, text="Photos", cursor="hand2", command=self.open_img, font=("Helvetica", 9, "bold"), fg="white",
                      bg="#42aad3", pady=10)
        b1_1.place(x=250, y=500, width=100, height=30)

        # Developer button
        img8 = Image.open(r"C:\\Users\\gagan\\Desktop\\FaceRecognition\\images\\student.png")
        img8 = img8.resize((100, 100), Image.LANCZOS)
        self.photoimg8 = ImageTk.PhotoImage(img8)

        b1 = Button(bg_img, image=self.photoimg8, cursor="hand2")
        b1.place(x=425, y=400, width=100, height=100)

        b1_1 = Button(bg_img, text="Developer", cursor="hand2", font=("Helvetica", 9, "bold"), fg="white",
                      bg="#42aad3", pady=10)
        b1_1.place(x=425, y=500, width=100, height=30)

        # Exit button
        img9 = Image.open(r"C:\\Users\\gagan\\Desktop\\FaceRecognition\\images\\student.png")
        img9 = img9.resize((100, 100), Image.LANCZOS)
        self.photoimg9 = ImageTk.PhotoImage(img9)

        b1 = Button(bg_img, image=self.photoimg9, cursor="hand2")
        b1.place(x=600, y=400, width=100, height=100)

        b1_1 = Button(bg_img, text="Exit", cursor="hand2", font=("Helvetica", 9, "bold"), fg="white",
                      bg="#42aad3", pady=10)
        b1_1.place(x=600, y=500, width=100, height=30)


    def open_img(self):
        os.startfile("data")


    # ============================================Fuction Button================================================

    def student_details(self):
        self.new_window=Toplevel(self.root)
        self.app=Student(self.new_window)



if __name__ == "__main__":
    root = Tk()
    obj = FaceRecognitionSystem(root)
    root.mainloop()
