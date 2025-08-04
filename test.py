import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Setup
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
labels = ["Hello", "How are you", "Fine", "Thank you", "Yes"]

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)
    imgOutput = img.copy()

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Crop the hand region
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        # Resize to square input
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal+hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        
        
        
        text = labels[index]
        font_scale = 1
        font_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        padding_x = 20
        padding_y = 20
        cv2.rectangle(
             imgOutput,
             (x, y - text_height - padding_y),  # top-left corner
              (x + text_width + padding_x, y),   # bottom-right corner
              (0, 0, 0),                         # black background
              -1                                # filled rectangle
        )
        cv2.putText(
            imgOutput,
            text,
            (x + 10, y - 10),                  # adjust text position inside box
             cv2.FONT_HERSHEY_SIMPLEX,
             font_scale,
             (255, 255, 255),                  # white text
             font_thickness
        )
        cv2.rectangle(
            imgOutput,
             (x - offset, y - offset),
             (x + w + offset, y + h + offset),
             (255, 0, 255),
             2
        )
        cv2.imshow("Cropped", imgCrop)
        cv2.imshow("White", imgWhite)
        cv2.imshow("Output", imgOutput)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                 break

        # Display
        #cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        #cv2.rectangle(imgOutput, (x, y - 60), (x + 150, y - 20), (0, 0, 0), -1)  # black bg for text
        #cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        #cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 2)
        #cv2.imshow("Cropped", imgCrop)
       # cv2.imshow("White", imgWhite)
    #cv2.imshow("Output", imgOutput)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        # break
   

cap.release()
cv2.destroyAllWindows()

