import cv2
import dlib

image = cv2.imread("fotos/grupo.0.jpg")
detector = dlib.get_frontal_face_detector()

facesDetectadas, pontuacao, idx = detector.run(image)



cv2.imshow("Detector Hog", image)
cv2.waitKey(0)
cv2.destroyAllWindows()