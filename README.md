#  Sign Language Detection

A real-time sign language recognition system using **Python**, **OpenCV**, and a pre-trained **Keras model** from **Teachable Machine**. It captures live webcam input, identifies hand gestures, and displays the predicted label on screen with clean visuals.


---

## What is it?

This project is a gesture recognition tool built to detect and classify sign language letters or commands using computer vision. It helps bridge the gap between spoken and signed communication, offering a lightweight offline prototype that runs on any laptop with a webcam.

---

## Key Features

-  **Real-Time Gesture Recognition**  
  Detects hand gestures in real time using webcam and bounding boxes.

-  **ML-Powered Predictions**  
  Uses a `.h5` Keras model trained with Teachable Machine to classify gestures.

- **Live Visualization**  
  Displays cropped and white-background images alongside output with bounding boxes and label overlays.

-  **Clean Label UI**  
  Shows detected labels in a black background box with white text for clarity.

- **Offline & Lightweight**  
  No internet required once model is downloaded — works entirely offline.

---

## Why I Built This?

I built this as part of my exploration into real-world computer vision applications using Python and deep learning. Sign language detection is an impactful problem that intersects accessibility, AI, and visual recognition — and I wanted to prototype a working tool using minimal resources and simple tech.

This project also helped me:
- Practice working with `OpenCV` and `cvzone`
- Understand how to preprocess input for ML models
- Learn how to convert models trained online (Teachable Machine) into usable `.h5` Keras models

---

##  What’s Next?

- Expand dataset for more gesture classes (A-Z or custom commands)
- Add frame smoothing to stabilize predictions
- Convert into a web-based Streamlit app (optional)
- Port to mobile or embedded hardware (like Raspberry Pi)
- Add guided training mode for users to record custom signs

---
