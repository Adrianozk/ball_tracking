
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# constrói os parâmetros que podem ser passados na hora de chamar o script
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="caminho para o arquivo de video (opcional) ")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="tamanho do buffer (rastro)")
args = vars(ap.parse_args())

# define os limites minimos e maximos
# para o threshold da bola no espaço de cor HSV
lower = (2, 139, 98)
upper = (33, 255, 197)

# inicializa a lista de pontos rastreados
pts = deque(maxlen=args["buffer"])

# se o caminho para o vídeo não foi disponibilizado,
# lê a referência para a webcam
if not args.get("video", False):
    video_stream = VideoStream(src=0).start()
# do contrário, pega a referência para o vídeo
else:
    video_stream = cv2.VideoCapture(args["video"])
# espera um tempo permitindo que a câmera ou o vídeo se "aqueçam"
time.sleep(2.0)

# loop infinito
while True:
    # lê o frame atual
    frame = video_stream.read()

    # manipula o frame por VideoCaptura ou VideStream
    frame = frame[1] if args.get("video", False) else frame
    # se você está vendo o vídeo e não capturou o frame,
    # então vc chegou a o final do vídeo
    if frame is None:
        break # fecha o loop

    # redimensiona o frame, desfoca , e converte para o espaço de cor HSV
    frame = imutils.resize(frame, width=600)
    # cv2.imshow("frame_original", frame) # só para "printar" as etapas
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    # cv2.imshow("gaussian_blur", blurred) # só para "printar" as etapas
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # constroe a máscara da cor da bola,
    # então executa uma série de dilatações e erosões
    # para remover quaisquer pequenas bolhas deixadas na máscara
    mask = cv2.inRange(hsv, lower, upper)

    # cv2.imshow("mask", mask) # só para "printar" as etapas

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # cv2.imshow("mask_with_erosion", mask) # só para "printar" as etapas

    # encontra contornos na máscara e inicializa o atual
    # (x, y) centro da bola
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # só continua se pelo menos um contorno foi encontrado
    if len(cnts) > 0:
        # encontra o maior contorno na mascara,
        # então o utiliza para calcular o círculo mínimo de fechamento e centróide
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # só continua se o raio bate com o tamanho mínimo
        if radius > 10:
            # desenha o círculo e centróide no frame
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    # atualiza a lista de pontos rastreados
    pts.appendleft(center)

    # loop sobe os pontos rastreados
    for i in range(1, len(pts)):
        # se qualquer dos pontos são None, ignora eles
        if pts[i - 1] is None or pts[i] is None:
            continue
        # do contrário, calcula a grossura da linha e
        # desenha as linhas conectadas
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), thickness)

    frame = np.flip(frame, 1)  # para inverter a imagem e ficar como espelho

    # mostra o frame na tela
    cv2.imshow("Projeto cabuloso", frame)

    key = cv2.waitKey(1) & 0xFF

    # se o botão 'q' é pressionado, para o loop
    if key == ord("q"):
        break

# cont = 0
# # for pt in pts:
# #     print("pts[" + str(cont) + "] = " + str(pt))
# #     cont = cont + 1

# se não está usando um arquivo de vídeo, para o fluxo de vídeo da câmera
if not args.get("video", False):
    video_stream.stop()
# do contrário, libera a câmera
else:
    video_stream.release()
# fecha todas as janelas
cv2.destroyAllWindows()
