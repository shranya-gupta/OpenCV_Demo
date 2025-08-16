# 🔍 OpenCv Multi-Mode System (OpenCV + Python)

## 📌 Project Overview
This project is an **OpenCV Multi-Mode System** built with **OpenCV** and **NumPy**.  
It allows you to switch between three different modes in real-time:

- **Normal Mode** → Displays your name, current mode, and system date/time.  
- **Contour Mode** → Detects objects based on contour area with adjustable thresholds using trackbars.  
- **Face Detection Mode** → Uses Haar Cascade Classifier to detect faces in the frame.  

The program works with a **webcam** but can also be easily adapted for an **image** or a **video file**.

---

## 🚀 Features
- 🎛️ Adjustable parameters for contour detection (`threshold1`, `threshold2`, `Area`) via Trackbars.  
- 😀 Face detection using Haar Cascade (`haarcascade_frontalface_default.xml`).  
- 📅 Overlay of **name + current date/time** in Normal Mode.  
- ⌨️ Interactive controls:
  - Press **`c`** → Contour mode  
  - Press **`f`** → Face detection mode  
  - Press **`n`** → Normal mode  
  - Press **`q`** → Quit  

---

## 🛠️ Technologies Used
- **Python 3.10+**
- **OpenCV**
- **NumPy**
- **Datetime**

---

