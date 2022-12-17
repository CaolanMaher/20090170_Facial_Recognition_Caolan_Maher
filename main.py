import cv2

video = cv2.VideoCapture(0)

faceDetect = cv2.CascadeClassifier()

while True:
    # capture the video
    ret,frame = video.read()

    # display the video
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()