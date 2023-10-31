# Save a video with different frame rates and specify the duration using OpenCV
import cv2
import time
from gaze_module import gaze_direction
from fer import *
import pickle
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#call gaze module
frames = gaze_direction(cap)
cv2.destroyAllWindows()

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

regressors = []
pcas = []
with open("fer_models\sdm_regressors_models.pckl", "rb") as f:
    while True:
        try:
            regressors.append(pickle.load(f))
        except EOFError:
            break
with open("fer_models\sdm_pcas_models.pckl", "rb") as f:
    while True:
        try:
            pcas.append(pickle.load(f))
        except EOFError:
            break

mean_landmarks = np.loadtxt("fer_models\mean_landmarks.txt")
fer_model = pickle.load(open('fer_models/fer_model.sav', 'rb'))

faces = get_faces(frames, face_classifier)
if len(faces)<1:
    print('no face')
else:
    intial_landmarks = np.asarray([np.round(mean_landmarks).astype(int) for _ in range(len(faces))])
    my_landmarks = test_landmarks(faces, intial_landmarks, regressors, pcas, [16,16,16,8,8,8,4,4,4])
    features = get_features(my_landmarks)

    motions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    predictions = fer_model.predict(features)
    start_time = time.time()
    for img, prediction, landmark in zip(faces, predictions, my_landmarks):
        for (x,y) in landmark:
            cv2.circle(img,(x,y),1,(0,255,0),2)
        cv2.imshow(motions[prediction] + ' Frame',img)
        while(True):
            if cv2.waitKey(5) & 0xFF == 27:
                cv2.destroyAllWindows()
                break
