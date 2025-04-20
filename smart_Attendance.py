import streamlit as st
import cv2
import face_recognition
import numpy as np
import sqlite3
import datetime
import os
import time

class AttendanceDatabase:
    def __init__(self, db_path='attendance.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        self.conn.execute('''CREATE TABLE IF NOT EXISTS users (
                                user_id TEXT PRIMARY KEY,
                                name TEXT NOT NULL,
                                roll_number TEXT NOT NULL,
                                encoding BLOB NOT NULL)''')
        self.conn.execute('''CREATE TABLE IF NOT EXISTS attendance (
                                user_id TEXT,
                                name TEXT,
                                roll_number TEXT,
                                timestamp TEXT)''')

    def add_user(self, user_id, name, roll_number, encoding):
        self.conn.execute("""INSERT OR REPLACE INTO users (user_id, name, roll_number, encoding) 
                             VALUES (?, ?, ?, ?)""", (user_id, name, roll_number, encoding))
        self.conn.commit()

    def delete_user(self, user_id):
        self.conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        self.conn.commit()

    def update_user(self, user_id, name, roll_number):
        self.conn.execute("UPDATE users SET name = ?, roll_number = ? WHERE user_id = ?",
                          (name, roll_number, user_id))
        self.conn.commit()

    def get_all_users(self):
        cursor = self.conn.execute("SELECT user_id, name, roll_number, encoding FROM users")
        return cursor.fetchall()

    def mark_attendance(self, user_id, name, roll_number):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conn.execute("INSERT INTO attendance (user_id, name, roll_number, timestamp) VALUES (?, ?, ?, ?)",
                          (user_id, name, roll_number, timestamp))
        self.conn.commit()

    def get_attendance_report(self):
        cursor = self.conn.execute("SELECT * FROM attendance")
        return cursor.fetchall()

    def get_attendance_by_date(self, date_str):
        cursor = self.conn.execute("SELECT user_id FROM attendance WHERE DATE(timestamp) = ?", (date_str,))
        return set(row[0] for row in cursor.fetchall())

    def get_attendance_dates(self):
        cursor = self.conn.execute("SELECT DISTINCT DATE(timestamp) FROM attendance ORDER BY DATE(timestamp) DESC")
        return [row[0] for row in cursor.fetchall()]

class FaceAttendanceSystem:
    def __init__(self):
        self.db = AttendanceDatabase()
        self.marked_ids = set()
        self.all_users = []
        self.attendance_running = False
        self.start_time = None

    def capture_face_encoding(self):
        cap = cv2.VideoCapture(0)
        encoding = None
        st.info("Capturing face. Please look at the camera. Press 'q' to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb_frame)
            if boxes:
                encodings = face_recognition.face_encodings(rgb_frame, boxes)
                if encodings:
                    encoding = encodings[0]
                    cv2.rectangle(frame, (boxes[0][3], boxes[0][0]), (boxes[0][1], boxes[0][2]), (0, 255, 0), 2)
            cv2.imshow("Face Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return encoding

    def register_user(self, user_id, name, roll_number):
        encoding = self.capture_face_encoding()
        if encoding is not None:
            self.db.add_user(user_id, name, roll_number, encoding.astype(np.float64).tobytes())
            st.success("User registered successfully!")
        else:
            st.error("Face capture failed.")

    def recognize_faces(self):
        self.all_users = self.db.get_all_users()
        if not self.all_users:
            st.warning("No students registered.")
            return

        known_encodings = [np.frombuffer(user[3], dtype=np.float64) for user in self.all_users]
        user_info = [(user[0], user[1], user[2]) for user in self.all_users]

        self.marked_ids = set()
        self.attendance_running = True
        self.start_time = time.time()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to access camera. Please check your webcam.")
            return

        st.info("Recognizing faces... Look at the camera. Press 'Stop Attendance' or wait 10 minutes.")

        while self.attendance_running:
            if time.time() - self.start_time > 600:
                break

            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb_frame)
            encodings = face_recognition.face_encodings(rgb_frame, boxes)

            for encoding in encodings:
                if not known_encodings:
                    continue

                matches = face_recognition.compare_faces(known_encodings, encoding)
                face_distances = face_recognition.face_distance(known_encodings, encoding)

                if len(face_distances) == 0:
                    continue

                best_match_index = np.argmin(face_distances)

                if matches and best_match_index < len(matches) and matches[best_match_index]:
                    user_id, name, roll_number = user_info[best_match_index]
                    user_id = str(user_id)
                    if user_id not in self.marked_ids:
                        self.db.mark_attendance(user_id, name, roll_number)
                        self.marked_ids.add(user_id)
                        st.success(f"✔️ Marked present: {name} ({roll_number})")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.attendance_running = False

    def stop_attendance(self):
        if self.attendance_running:
            self.attendance_running = False
            st.info("Attendance manually stopped.")

app = FaceAttendanceSystem()
st.title("AI-Based Smart Face Attendance System")

menu = st.sidebar.selectbox("Select Option", [
    "Register Student", "Take Attendance", "View Attendance Report", 
    "Delete Student", "Update Student", "View All Students"])

if menu == "Register Student":
    with st.form("register_form"):
        user_id = st.text_input("Student ID")
        name = st.text_input("Student Name")
        roll_number = st.text_input("Roll Number")
        submit = st.form_submit_button("Add Student")
        if submit:
            app.register_user(user_id, name, roll_number)

elif menu == "Take Attendance":
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Attendance") and not app.attendance_running:
            app.recognize_faces()
    with col2:
        if st.button("Stop Attendance"):
            app.stop_attendance()

elif menu == "View Attendance Report":
    st.subheader("Select Date to View Report")
    dates = app.db.get_attendance_dates()
    if dates:
        selected_date = st.selectbox("Select Date", dates)
    else:
        selected_date = datetime.datetime.now().strftime("%Y-%m-%d")
        st.info("No attendance data found. Showing today by default.")

    present_ids = app.db.get_attendance_by_date(selected_date)
    all_users = app.db.get_all_users()

    present_students = []
    absent_students = []

    for user_id, name, roll_number, _ in all_users:
        if user_id in present_ids:
            present_students.append((user_id, name, roll_number))
        else:
            absent_students.append((user_id, name, roll_number))

    st.subheader("Present Students")
    if present_students:
        for student in present_students:
            st.success(f"✔️ {student[1]} ({student[2]})")
    else:
        st.info("No students were marked present on this date.")

    st.subheader("Absent Students")
    if absent_students:
        for student in absent_students:
            st.error(f"❌ {student[1]} ({student[2]})")
    else:
        st.info("No absent students on this date.")

    if st.button("Download Attendance Report"):
        data = app.db.get_attendance_report()
        report_path = "attendance_report.csv"
        with open(report_path, "w") as f:
            f.write("User ID,Name,Roll Number,Timestamp\n")
            for row in data:
                f.write(",".join(row) + "\n")
        st.download_button("Download Report", data=open(report_path, "rb"), file_name="attendance_report.csv")

elif menu == "Delete Student":
    user_id = st.text_input("Enter Student ID to Delete")
    if st.button("Delete"):
        app.db.delete_user(user_id)
        st.success("Student deleted successfully.")

elif menu == "Update Student":
    with st.form("update_form"):
        user_id = st.text_input("Student ID")
        name = st.text_input("New Name")
        roll_number = st.text_input("New Roll Number")
        submit = st.form_submit_button("Update Student")
        if submit:
            app.db.update_user(user_id, name, roll_number)
            st.success("Student updated successfully.")

elif menu == "View All Students":
    users = app.db.get_all_users()
    if users:
        st.table([(u[0], u[1], u[2]) for u in users])
    else:
        st.info("No students registered yet.")