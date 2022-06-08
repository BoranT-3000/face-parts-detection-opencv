import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
smileCascade = cv2.CascadeClassifier("haarcascade_smile.xml")


color = {"blue": (255, 0, 0), "red": (0, 0, 255),
         "green": (0, 255, 0), "white": (255, 255, 255),
         "magenta": (255, 0, 255), "aqua": (0, 255, 255),
         "Black": (0, 0, 0), "yellow": (255, 255, 0),
         "deep_sky_blue": (0, 191, 255)}


def selfie(name):
    print("Image "+str(name)+".jpg is Saved")
    path = str(name)+'.jpg'
    cv2.imwrite(path, img)


cap = cv2.VideoCapture(0)

while True:
    # reading image from web-cam
    _, img = cap.read()  # ret

    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)

    # faces = face_cascade.detectMultiScale(gray_frame, 1.1, 10)
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    count = 1

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), color["red"], 2)
        cv2.putText(img, "Face", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color["red"], 1, cv2.LINE_AA)
        # center = (x + w//2, y + h//2)
        # frame = cv2.ellipse(img, center, (w//2, h//2),
        #                     0, 0, 360, (255, 0, 255), 4)

        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=12,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 12)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey),
                          (ex+ew, ey+eh), color["green"], 2)
            cv2.putText(roi_color, "Eye", (ex, ey-4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color["green"], 1, cv2.LINE_AA)

        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.1, 20)
        # mouth = mouth_cascade.detectMultiScale(
        #     gray_frame, scaleFactor=1.05, minNeighbors=5,
        #     minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (mx, my, mw, mh) in mouth:
            cv2.rectangle(roi_color, (mx, my),
                          (mx+mw, my+mh), color["white"], 2)
            cv2.putText(roi_color, "Mouth", (mx, my-4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color["white"], 1, cv2.LINE_AA)
            count += 1
            if(count >= 2):
                break

        nose = nose_cascade.detectMultiScale(roi_gray, 1.1, 4)
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(roi_color, (nx, ny),
                          (nx+nw, ny+nh), color["blue"], 2)
            cv2.putText(roi_color, "Nose", (nx, ny-4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color["blue"], 1, cv2.LINE_AA)
            count += 1
            if(count >= 2):
                break

        smiles = smileCascade.detectMultiScale(roi_gray, 1.8, 15)
        for sx, sy, sw, sh in smiles:
            cv2.rectangle(roi_color, (sx, sy),
                          (sx+sw, sy+sh), color["yellow"], 5)
            label = "smile"
            cv2.putText(roi_color, "Smile", (sx, sy-4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color["yellow"], 2, cv2.LINE_AA)

            # selfie("smile")

            count += 1
            if(count >= 2):
                break

    cv2.imshow('img', img)
    if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.waitKey(30) & 0xff == 27):
        break

# it relases web-cam
cap.release()
# this command destroys output window
cv2.destroyAllWindows()
