from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps, ImageDraw
from tkinter import messagebox
import mysql.connector
import cv2

class Student:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1920x1200+0+0")
        self.root.title("Students Details")

        # =================================variables=====================================
        self.var_dep = StringVar()
        self.var_course = StringVar()
        self.var_year = StringVar()
        self.var_semester = StringVar()
        self.var_std_id = StringVar()
        self.var_std_name = StringVar()
        self.var_div = StringVar()
        self.var_roll = StringVar()
        self.var_gender = StringVar()
        self.var_dob = StringVar()
        self.var_email = StringVar()
        self.var_phone = StringVar()
        self.var_address = StringVar()
        self.var_teacher = StringVar()

        img = Image.open(r"C:\Users\gagan\Desktop\FaceRecognition\images\face.jpg")
        self.photoimg = ImageTk.PhotoImage(img)

        bg_img = Label(self.root, image=self.photoimg)
        bg_img.place(x=0, y=0, width=1920, relheight=1)

        title_lbl = Label(bg_img, text="STUDENTS MANAGEMENT SYSTEM", font=("Helvetica", 20, "bold"), fg="white",
                          bg="#1b1b33", pady=10)
        title_lbl.place(x=273, y=100, anchor="w")

        main_frame = Frame(bg_img, bd=2, bg="#21618e")
        main_frame.place(x=20, y=150, width=1230, height=500)

        # left label frame
        Left_frame = LabelFrame(main_frame, bd=2, bg="#04385f", relief=RAISED, text="Students Details",
                                font=("Helvetica", 12, "bold"), fg="white")
        Left_frame.place(x=10, y=10, width=680, height=470)

        # current course information
        current_course = LabelFrame(Left_frame, bd=2, bg="#04385f", relief=RAISED, text="Current course information",
                                    font=("Helvetica", 12, "bold"), fg="white")
        current_course.place(x=10, y=10, width=660, height=125)

        # department
        dep_label = Label(current_course, text="Department", font=("Helvetica", 12, "bold"), fg="white", bg="#36607f")
        dep_label.grid(row=0, column=0, padx=10)

        dep_combo = ttk.Combobox(current_course, textvariable=self.var_dep, font=("Helvetica", 12, "bold"), width=17,
                                 state="readonly")
        dep_combo["values"] = (
            "Select Department", "Computer", "Data Science", "AI/ML", "IOT", "Mechanical", "Electrical")
        dep_combo.current(0)
        dep_combo.grid(row=0, column=1, padx=2, pady=10)

        # Course
        course_label = Label(current_course, text="Course", font=("Helvetica", 12, "bold"), fg="white", bg="#36607f")
        course_label.grid(row=0, column=2, padx=10, sticky=W)

        course_label = ttk.Combobox(current_course, textvariable=self.var_course, font=("Helvetica", 12, "bold"),
                                    width=17, state="readonly")
        course_label["values"] = (
            "Select Course", "FE", "SE", "TE", "BE")
        course_label.current(0)
        course_label.grid(row=0, column=3, padx=2, pady=10, sticky=W)

        # Year
        year_label = Label(current_course, text="Year", font=("Helvetica", 12, "bold"), fg="white", bg="#36607f")
        year_label.grid(row=1, column=0, padx=10, sticky=W)

        year_combo = ttk.Combobox(current_course, textvariable=self.var_year, font=("Helvetica", 12, "bold"), width=17,
                                  state="read only")
        year_combo["values"] = (
            "Select Year", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
        year_combo.current(0)
        year_combo.grid(row=1, column=1, padx=2, pady=10, sticky=W)

        # semester

        semester_label = Label(current_course, text="Semester", font=("Helvetica", 12, "bold"), fg="white",
                               bg="#36607f")
        semester_label.grid(row=1, column=2, padx=10, sticky=W)

        semester_combo = ttk.Combobox(current_course, textvariable=self.var_semester, font=("Helvetica", 12, "bold"),
                                      width=17, state="read only")
        semester_combo["values"] = (
            "Select Semester", "Semester-1", "Semester-2", "Semester-3", "Semester-4", "Semester-5", "Semester-6",
            "Semester-7", "Semester-8")
        semester_combo.current(0)
        semester_combo.grid(row=1, column=3, padx=2, pady=10, sticky=W)

        # class student information
        class_student_frame = LabelFrame(Left_frame, bd=2, bg="#04385f", relief=RAISED,
                                         text="Class Students Information",
                                         font=("Helvetica", 12, "bold"), fg="white")
        class_student_frame.place(x=10, y=135, width=660, height=300)

        # student id
        studentId_label = Label(class_student_frame, text="StudentID", font=("Helvetica", 12, "bold"), fg="white",
                                bg="#36607f")
        studentId_label.grid(row=0, column=0, padx=10, sticky=W)

        studentID_entry = ttk.Entry(class_student_frame, textvariable=self.var_std_id, width=20,
                                    font=("Helvetica", 12, "bold"))
        studentID_entry.grid(row=0, column=1, padx=10, sticky=W)

        # student name
        studentName_label = Label(class_student_frame, text="Student Name", font=("Helvetica", 12, "bold"), fg="white",
                                  bg="#36607f")
        studentName_label.grid(row=0, column=2, padx=8, sticky=W)

        studentName_entry = ttk.Entry(class_student_frame, textvariable=self.var_std_name, width=20,
                                      font=("Helvetica", 12, "bold"))
        studentName_entry.grid(row=0, column=3, padx=8, sticky=W)

        # class division
        class_div_label = Label(class_student_frame, text="Class Division", font=("Helvetica", 12, "bold"), fg="white",
                                bg="#36607f")
        class_div_label.grid(row=1, column=0, padx=10, pady=5, sticky=W)

        div_combo = ttk.Combobox(class_student_frame, textvariable=self.var_div, font=("Helvetica", 12, "bold"),
                                 width=10,
                                 state="read only")
        div_combo["values"] = (
            "Select", "A", "B", "C", "D", "E", "F", "G", "H")
        div_combo.current(0)
        div_combo.grid(row=1, column=1, padx=10, pady=5, sticky=W)

        # Roll No
        roll_no_label = Label(class_student_frame, text="Roll No", font=("Helvetica", 12, "bold"), fg="white",
                              bg="#36607f")
        roll_no_label.grid(row=1, column=2, padx=10, pady=5, sticky=W)

        roll_no_entry = ttk.Entry(class_student_frame, textvariable=self.var_roll, width=20,
                                  font=("Helvetica", 12, "bold"))
        roll_no_entry.grid(row=1, column=3, padx=10, pady=5, sticky=W)

        # Gender
        gender_label = Label(class_student_frame, text="Gender", font=("Helvetica", 12, "bold"), fg="white",
                             bg="#36607f")
        gender_label.grid(row=2, column=0, padx=10, pady=5, sticky=W)

        gender_combo = ttk.Combobox(class_student_frame, textvariable=self.var_gender, font=("Helvetica", 12, "bold"),
                                    width=10,
                                    state="read only")
        gender_combo["values"] = (
            "Select", "Male", "Female", "Other")
        gender_combo.current(0)
        gender_combo.grid(row=2, column=1, padx=10, pady=5, sticky=W)

        # Dob
        dob_label = Label(class_student_frame, text="DOB", font=("Helvetica", 12, "bold"), fg="white",
                          bg="#36607f")
        dob_label.grid(row=2, column=2, padx=10, pady=5, sticky=W)

        dob_entry = ttk.Entry(class_student_frame, textvariable=self.var_dob, width=20, font=("Helvetica", 12, "bold"))
        dob_entry.grid(row=2, column=3, padx=10, pady=5, sticky=W)

        # Email
        email_label = Label(class_student_frame, text="Email", font=("Helvetica", 12, "bold"), fg="white",
                            bg="#36607f")
        email_label.grid(row=3, column=0, padx=10, pady=5, sticky=W)

        email_entry = ttk.Entry(class_student_frame, textvariable=self.var_email, width=20,
                                font=("Helvetica", 12, "bold"))
        email_entry.grid(row=3, column=1, padx=10, pady=5, sticky=W)

        # Phone no
        phone_label = Label(class_student_frame, text="Phone No", font=("Helvetica", 12, "bold"), fg="white",
                            bg="#36607f")
        phone_label.grid(row=3, column=2, padx=10, pady=5, sticky=W)

        phone_entry = ttk.Entry(class_student_frame, textvariable=self.var_phone, width=20,
                                font=("Helvetica", 12, "bold"))
        phone_entry.grid(row=3, column=3, padx=10, pady=5, sticky=W)

        # Address
        address_label = Label(class_student_frame, text="Address", font=("Helvetica", 12, "bold"), fg="white",
                              bg="#36607f")
        address_label.grid(row=4, column=0, padx=10, pady=5, sticky=W)

        address_entry = ttk.Entry(class_student_frame, textvariable=self.var_address, width=20,
                                  font=("Helvetica", 12, "bold"))
        address_entry.grid(row=4, column=1, padx=10, pady=5, sticky=W)

        # Teacher name
        teacher_label = Label(class_student_frame, text="Teacher", font=("Helvetica", 12, "bold"), fg="white",
                              bg="#36607f")
        teacher_label.grid(row=4, column=2, padx=10, pady=5, sticky=W)

        teacher_entry = ttk.Entry(class_student_frame, textvariable=self.var_teacher, width=20,
                                  font=("Helvetica", 12, "bold"))
        teacher_entry.grid(row=4, column=3, padx=10, pady=5, sticky=W)

        # radio buttons
        self.var_radio1 = StringVar()
        radiobtn1 = ttk.Radiobutton(class_student_frame, variable=self.var_radio1, text="Take Photo Sample",
                                    value="YES")
        radiobtn1.grid(row=5, column=0)

        # photo sample radio buttons
        radiobtn2 = ttk.Radiobutton(class_student_frame, variable=self.var_radio1, text="No Photo Sample", value="NO")
        radiobtn2.grid(row=5, column=1)

        # Button frame
        btn_frame = Frame(class_student_frame, bd=2, relief=RIDGE, bg="#04385f")
        btn_frame.place(x=5, y=200, width=644, height=35)

        save_btn = Button(btn_frame, text="Save", command=self.add_data, width=15, font=("Helvetica", 12, "bold"),
                          fg="white",
                          bg="#3bc55e")
        save_btn.grid(row=0, column=0)

        update_btn = Button(btn_frame, text="Update", command=self.update_function, width=15,
                            font=("Helvetica", 12, "bold"), fg="white",
                            bg="#ddda36")
        update_btn.grid(row=0, column=1)

        delete_btn = Button(btn_frame, text="Delete", command=self.delete_data, width=15,
                            font=("Helvetica", 12, "bold"), fg="white", bg="#c53b52")
        delete_btn.grid(row=0, column=2)

        reset_btn = Button(btn_frame, text="Reset", command=self.reset_data, width=15, font=("Helvetica", 12, "bold"),
                           fg="white", bg="#be253f")
        reset_btn.grid(row=0, column=3)

        btn_frame1 = Frame(class_student_frame, bd=2, relief=RIDGE, bg="#04385f")
        btn_frame1.place(x=5, y=235, width=644, height=38)

        take_photo_btn = Button(btn_frame1, text="Take Photo Sample", command=self.generate_dataset, width=31, font=("Helvetica", 12, "bold"),
                                fg="white", bg="#3c0f4c")
        take_photo_btn.grid(row=1, column=0)

        update_photo_btn = Button(btn_frame1, text="Update Photo Sample", width=31, font=("Helvetica", 12, "bold"),
                                  fg="white", bg="#3c0f4c")
        update_photo_btn.grid(row=1, column=1)

        # right label frame
        right_frame = LabelFrame(main_frame, bd=2, bg="#04385f", relief=RAISED, text="Students Details",
                                 font=("Helvetica", 12, "bold"), fg="white")
        right_frame.place(x=695, y=10, width=520, height=470)

        # ===================================SEARCHING SYSTEM=============================================

        search_frame = LabelFrame(right_frame, bd=2, bg="#04385f", relief=RAISED, text="Search System",
                                  font=("Helvetica", 12, "bold"), fg="white")
        search_frame.place(x=0, y=0, width=515, height=70)

        search_label = Label(search_frame, text="Search By", font=("Helvetica", 12, "bold"), fg="white", bg="#36607f")
        search_label.grid(row=0, column=0, padx=10, pady=5, sticky=W)

        search_combo = ttk.Combobox(search_frame, font=("Helvetica", 12, "bold"), width=10, state="readonly")
        search_combo["values"] = ("Select", "Roll No", "Phone No")
        search_combo.current(0)
        search_combo.grid(row=0, column=1, padx=2, pady=10, sticky=W)

        search_entry = ttk.Entry(search_frame, width=10, font=("Helvetica", 12, "bold"))
        search_entry.grid(row=0, column=2, padx=10, pady=5, sticky=W)

        search_btn = Button(search_frame, text="Search", width=7, font=("Helvetica", 12, "bold"), fg="white",
                            bg="#FFC96F")
        search_btn.grid(row=0, column=3, padx=3)

        showAll_btn = Button(search_frame, text="Show All", width=7, font=("Helvetica", 12, "bold"), fg="white",
                             bg="#FFC96F")
        showAll_btn.grid(row=0, column=4, padx=3)

        # ====================================table frame==============================================

        table_frame = Frame(right_frame, bd=2, bg="#04385f", relief=RAISED)
        table_frame.place(x=0, y=75, width=515, height=369)

        scroll_x = ttk.Scrollbar(table_frame, orient=HORIZONTAL)
        scroll_y = ttk.Scrollbar(table_frame, orient=VERTICAL)

        self.student_table = ttk.Treeview(table_frame, column=(
            "dep", "course", "year", "sem", "id", "name", "div", "roll", "gender", "dob", "email", "phone", "address",
            "teacher", "photo"), xscrollcommand=scroll_x, yscrollcommand=scroll_y)

        scroll_x.pack(side=BOTTOM, fill=X)
        scroll_y.pack(side=RIGHT, fill=Y)
        scroll_x.config(command=self.student_table.xview)
        scroll_y.config(command=self.student_table.yview)

        self.student_table.heading("dep", text="Department")
        self.student_table.heading("course", text="Course")
        self.student_table.heading("year", text="Year")
        self.student_table.heading("sem", text="Semester")
        self.student_table.heading("id", text="StudentID")
        self.student_table.heading("name", text="Name")
        self.student_table.heading("div", text="Division")
        self.student_table.heading("roll", text="Roll No")
        self.student_table.heading("gender", text="Gender")
        self.student_table.heading("dob", text="DOB")
        self.student_table.heading("email", text="Email")
        self.student_table.heading("phone", text="Phone")
        self.student_table.heading("address", text="Address")
        self.student_table.heading("teacher", text="Teacher")
        self.student_table.heading("photo", text="PhotoSampleStatus")
        self.student_table["show"] = "headings"

        self.student_table.column("dep", width=100)
        self.student_table.column("course", width=100)
        self.student_table.column("year", width=100)
        self.student_table.column("sem", width=100)
        self.student_table.column("id", width=100)
        self.student_table.column("name", width=100)
        self.student_table.column("div", width=100)
        self.student_table.column("roll", width=100)
        self.student_table.column("gender", width=100)
        self.student_table.column("dob", width=100)
        self.student_table.column("email", width=100)
        self.student_table.column("phone", width=100)
        self.student_table.column("address", width=100)
        self.student_table.column("teacher", width=100)
        self.student_table.column("photo", width=100)

        self.student_table.pack(fill=BOTH, expand=1)
        self.student_table.bind("<ButtonRelease>", self.get_cursor)
        self.fetch_data()

    # ===================================Function Declaration============================================================

    def add_data(self):
        if self.var_dep.get() == "Select Department" or self.var_std_name.get() == "" or self.var_std_id.get() == "":
            messagebox.showerror("Error", "All fields are required", parent=self.root)
        else:
            try:
                conn = mysql.connector.connect(host="localhost", username="root", password="3638",
                                               database="face_recognizer")
                my_cursor = conn.cursor()
                my_cursor.execute("insert into student values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (
                    self.var_dep.get(),
                    self.var_course.get(),
                    self.var_year.get(),
                    self.var_semester.get(),
                    self.var_std_id.get(),
                    self.var_std_name.get(),
                    self.var_div.get(),
                    self.var_roll.get(),
                    self.var_gender.get(),
                    self.var_dob.get(),
                    self.var_email.get(),
                    self.var_phone.get(),
                    self.var_address.get(),
                    self.var_teacher.get(),
                    self.var_radio1.get(),
                ))
                conn.commit()
                self.fetch_data()
                conn.close()
                messagebox.showinfo("Success", "Student details has been added successfully", parent=self.root)
            except Exception as es:
                messagebox.showerror("Error", f"Due To :{str(es)}", parent=self.root)

    # ================================fetch data=================================
    def fetch_data(self):
        conn = mysql.connector.connect(host="localhost", username="root", password="3638", database="face_recognizer")
        my_cursor = conn.cursor()
        my_cursor.execute("select * from student")
        data = my_cursor.fetchall()

        if len(data) != 0:
            self.student_table.delete(*self.student_table.get_children())
            for i in data:
                self.student_table.insert("", END, values=i)
            conn.commit()
        conn.close()

    # =========================get cursor=========================
    def get_cursor(self, event=""):
        cursor_focus = self.student_table.focus()
        content = self.student_table.item(cursor_focus)
        data = content["values"]

        self.var_dep.set(data[0]),
        self.var_course.set(data[1]),
        self.var_year.set(data[2]),
        self.var_semester.set(data[3]),
        self.var_std_id.set(data[4]),
        self.var_std_name.set(data[5]),
        self.var_div.set(data[6]),
        self.var_roll.set(data[7]),
        self.var_gender.set(data[8]),
        self.var_dob.set(data[9]),
        self.var_email.set(data[10]),
        self.var_phone.set(data[11]),
        self.var_address.set(data[12]),
        self.var_teacher.set(data[13]),
        self.var_radio1.set(data[14]),

    # ===================================update function===================================================
    def update_function(self):
        if self.var_dep.get() == "Select Department" or self.var_std_name.get() == "" or self.var_std_id.get() == "":
            messagebox.showerror("Error", "All fields are required", parent=self.root)
        else:
            try:
                update = messagebox.askyesno("update", "Do you want to update this student details", parent=self.root)
                if update > 0:
                    conn = mysql.connector.connect(host="localhost", username="root", password="3638",
                                                   database="face_recognizer")
                    my_cursor = conn.cursor()
                    my_cursor.execute(
                        "update student set Dep=%s,course=%s,Year=%s,Semester=%s,Name=%s,Division=%s,Roll=%s,Gender=%s,Dob=%s,Email=%s,Phone=%s,Address=%s,Teacher=%s,PhotoSample=%s where Student_id=%s",
                        (
                            self.var_dep.get(),
                            self.var_course.get(),
                            self.var_year.get(),
                            self.var_semester.get(),
                            self.var_std_name.get(),
                            self.var_div.get(),
                            self.var_roll.get(),
                            self.var_gender.get(),
                            self.var_dob.get(),
                            self.var_email.get(),
                            self.var_phone.get(),
                            self.var_address.get(),
                            self.var_teacher.get(),
                            self.var_radio1.get(),
                            self.var_std_id.get()
                        ))

                else:
                    if not update:
                        return

                messagebox.showinfo("Success", "Students details successfully update completed", parent=self.root)
                conn.commit()
                self.fetch_data()
                conn.close()
            except Exception as es:
                messagebox.showerror("Error", f"Due To:{str(es)}", parent=self.root)

    #   ================================delete_function===============================================

    def delete_data(self):
        if self.var_std_id.get() == "":
            messagebox.showerror("Error", "Student id must be required", parent=self.root)
        else:
            try:
                delete = messagebox.askyesno("Delete Student", "Do you want to delete this student", parent=self.root)
                if delete > 0:
                    conn = mysql.connector.connect(host="localhost", username="root", password="3638",
                                                   database="face_recognizer")
                    my_cursor = conn.cursor()
                    sql = "delete from student where Student_id=%s"
                    val = (self.var_std_id.get(),)
                    my_cursor.execute(sql, val)
                else:
                    if not delete:
                        return
                conn.commit()
                self.fetch_data()
                conn.close()

                messagebox.showinfo("Delete", "Successfully deleted student", parent=self.root)

            except Exception as es:
                messagebox.showerror("Error", f"Due To:{str(es)}", parent=self.root)

    # =======================================Reset Button=====================================
    def reset_data(self):
        self.var_dep.set("Select Department")
        self.var_course.set("Select Course")
        self.var_year.set("Select Year")
        self.var_semester.set("Select Semester")
        self.var_std_id.set("")
        self.var_std_name.set("")
        self.var_div.set("Select")
        self.var_roll.set("Select")
        self.var_gender.set("Select")
        self.var_dob.set("")
        self.var_email.set("")
        self.var_phone.set("")
        self.var_address.set("")
        self.var_teacher.set("")
        self.var_radio1.set("")


# ==================================Generate data set or take photo samples===============================
    def generate_dataset(self):
        if self.var_dep.get() == "Select Department" or self.var_std_name.get() == "" or self.var_std_id.get() == "":
            messagebox.showerror("Error", "All fields are required", parent=self.root)
        else:
            try:
                conn = mysql.connector.connect(host="localhost", username="root", password="3638", database="face_recognizer")
                my_cursor = conn.cursor()
                my_cursor.execute("select * from student")
                myresult = my_cursor.fetchall()
                id = len(myresult) + 1  # Get the next available ID

                # Update student data in the database
                my_cursor.execute(
                    "update student set Dep=%s, course=%s, Year=%s, Semester=%s, Name=%s, Division=%s, Roll=%s, Gender=%s, Dob=%s, Email=%s, Phone=%s, Address=%s, Teacher=%s, PhotoSample=%s where Student_id=%s",
                    (
                        self.var_dep.get(),
                        self.var_course.get(),
                        self.var_year.get(),
                        self.var_semester.get(),
                        self.var_std_name.get(),
                        self.var_div.get(),
                        self.var_roll.get(),
                        self.var_gender.get(),
                        self.var_dob.get(),
                        self.var_email.get(),
                        self.var_phone.get(),
                        self.var_address.get(),
                        self.var_teacher.get(),
                        self.var_radio1.get(),
                        id  # Use the updated ID here
                    ))
                conn.commit()
                self.fetch_data()
                self.reset_data()
                conn.close()

                # Load predefined data on face frontals from OpenCV
                face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

                def face_cropped(img):
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
                    if len(faces) == 0:
                        return None  # No face detected
                    (x, y, w, h) = faces[0]  # Assuming only one face is detected
                    face_cropped = img[y:y+h, x:x+w]
                    return face_cropped

                cap = cv2.VideoCapture(0)
                img_id = 0
                while True:
                    ret, my_frame = cap.read()
                    if not ret:
                        break  # If reading from the camera fails, break the loop
                    cropped_face = face_cropped(my_frame)
                    if cropped_face is not None:
                        img_id += 1
                        face = cv2.resize(cropped_face, (450, 450))
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        file_name_path = f"data/user.{id}.{img_id}.jpg"
                        cv2.imwrite(file_name_path, face)
                        cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                        cv2.imshow("Cropped Face", face)
                    if cv2.waitKey(1) == 13 or img_id == 100:
                        break
                cap.release()
                cv2.destroyAllWindows()
                messagebox.showinfo("Result", "Generating data set completed!!!!")

            except Exception as es:
                messagebox.showerror("Error", f"Due To: {str(es)}", parent=self.root)






if __name__ == "__main__":
    root = Tk()
    obj = Student(root)
    root.mainloop()
