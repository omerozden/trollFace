import cv2
img1 = cv2.imread("troll.png")
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while(True):
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    for (x,y,w,h) in faces:
        resized_image = cv2.resize(img1, (h, w))
        rows, cols, channels = resized_image.shape
        roi = frame[y:y + rows, x:x + cols]
        img2gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img2_fg = cv2.bitwise_and(resized_image, resized_image, mask=mask)
        dst = cv2.add(img1_bg, img2_fg)
        frame[y:y + rows, x:x + cols] = dst
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
