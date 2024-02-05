import cv2
import dlib

subdetector = ["Olhar a frente", "Vista a esquerda", "Vista a Direita", "Vista a frente girando a esquerda", "Vista a frente girando a direita"]

image = cv2.imread("fotos/grupo.0.jpg")
detector = dlib.get_frontal_face_detector()
facesDetectadas, pontuacao, idx = detector.run(image, 1, 0) #1 é a escala da imagem que pode ser modificada   #0 filtro

print(facesDetectadas)
print(pontuacao)
print(idx)

for i , d in enumerate(facesDetectadas):
    print("Detecção: {}, pontuação: {}, Sub-detector: {}".format(d, pontuacao[i], subdetector[(idx[i])]))
    e, t, d, b = (int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))
    cv2.rectangle(image, (e,t), (d, b) , (0 , 0, 255), 2)


cv2.imshow("Detector Hog", image)
cv2.waitKey(0)
cv2.destroyAllWindows()