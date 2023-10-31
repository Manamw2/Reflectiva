import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.decomposition import PCA
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#import seaborn as sns

import cv2
import PIL.Image, PIL.ImageTk

def process_images(folder_path: str):
    """
    Process images from the given folder path and return a list of processed images.
    :param folder_path: folder path
    :return: list of processed images
    """
    # get all the images from the folder
    images = [cv2.imread(folder_path + '/' + image)
              for image in os.listdir(folder_path)]
    #images = [cv2.resize(image, (250,250)) for image in images]
    # convert the images to RGB
    #images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    return images

def get_faces(images, face_classifier):
  train_images = []

  for i,image in enumerate(images):
    # Get faces into webcam's image
    #rects = detector(image, 0)
    face = face_classifier.detectMultiScale(
    image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(face)>0:
      (x, y, w, h) = face[0]
      #get face
      #(x, y, w, h) = face_utils.rect_to_bb(rects[0])
      image = image[y:y+h, x:x+w]
      image = cv2.resize(image, (256, 256))

      train_images.append(image)
  return train_images

def extract_images_sift(images, images_landmarks, patch_size):
  sift = cv2.SIFT_create()
  images_sift_descriptors = []
  for image, image_landmarks in zip(images, images_landmarks):
    image_descriptors = []
    for landmarks_xy in image_landmarks:
        x,y = landmarks_xy[0], landmarks_xy[1]
        sift_points = []
        cv_landmark = cv2.KeyPoint(float(x), float(y), patch_size)

        sift_points += [cv_landmark, cv_landmark]
        _, descriptors = sift.compute(image, sift_points)
        image_descriptors.append(descriptors)
    images_sift_descriptors.append(image_descriptors)
  images_sift_descriptors = np.asarray(images_sift_descriptors)
  return images_sift_descriptors

def test_landmarks(images, intial_landmarks, regressors, pcas, patches):
  images_current_landmarks = intial_landmarks
  for lr, pca, patch_size in zip(regressors, pcas, patches):
    images_sift_descriptors = extract_images_sift(images, images_current_landmarks, patch_size)
    num_images, num_landmarks, num_landmark_coordinates, descriptor_size = images_sift_descriptors.shape
    descriptors = images_sift_descriptors.reshape(num_images,num_landmarks*num_landmark_coordinates*descriptor_size)
    descriptors = pca.transform(descriptors)
    predicted_landmarks_diff = lr.predict(descriptors)
    predicted_landmarks_diff = predicted_landmarks_diff.reshape(num_images,num_landmarks,num_landmark_coordinates)
    images_current_landmarks = images_current_landmarks + predicted_landmarks_diff.astype(int)
  return images_current_landmarks

def get_features(train_landmarks):
  feature_vectors = []

  for landmarks in train_landmarks:
    nose_angle = math.atan((landmarks[27][1]-landmarks[30][1])
                          / (landmarks[27][0]-landmarks[30][0] + 10**-7))
    X_cog = np.mean(landmarks[:,0])
    Y_cog = np.mean(landmarks[:,1])
    X_relative = [i-X_cog for i in landmarks[:,0]]
    Y_relative = [i-Y_cog for i in landmarks[:,1]]
    EUC = [np.sqrt(np.square(i-X_cog) + np.square(j-Y_cog)) for (i,j) in landmarks]
    theta =  [math.atan((j - Y_cog) / (i - X_cog + 10**-7)) - nose_angle for (i,j) in landmarks]

    v = X_relative + Y_relative + EUC + theta
    feature_vectors.append(v)
  return feature_vectors

def fit(images, true_landmarks):
  images_current_landmarks = np.asarray([np.mean(true_landmarks, axis=0).astype(int) for _ in range(len(true_landmarks))])
  patch_size = 16
  regressors = []
  pcas = []
  for regressor_id in range(9):
    print("finding the sift descriptors")
    print('patch size = ', patch_size)
    images_sift_descriptors = extract_images_sift(images, images_current_landmarks, patch_size)
    #patch_size -= 4
    if regressor_id == 2:
      patch_size = 8
    if regressor_id == 5:
      patch_size = 4
    num_images, num_landmarks, num_landmark_coordinates, descriptor_size = images_sift_descriptors.shape
    descriptors = images_sift_descriptors.reshape(num_images,num_landmarks*num_landmark_coordinates*descriptor_size)
    print("getting PCA")
    pca = PCA(n_components=0.97)
    pca.fit(descriptors)
    pcas.append(pca)
    descriptors = pca.transform(descriptors)
    print("PCA is applied on the SIFT descriptors!")
    images_landmarks_diff = true_landmarks - images_current_landmarks
    images_landmarks_diff = images_landmarks_diff.reshape(num_images,num_landmarks*num_landmark_coordinates)
    #print(images_delta_landmarks)
    print(f"Training regressor num ",regressor_id)
    lr = LinearRegression()
    lr.fit(descriptors, images_landmarks_diff)
    regressors.append(lr)
    # predicting the diffrence between true and predicted landmarks
    predicted_landmarks_diff = lr.predict(descriptors)
    predicted_landmarks_diff = predicted_landmarks_diff.reshape(num_images,num_landmarks,num_landmark_coordinates)
    images_current_landmarks = images_current_landmarks + predicted_landmarks_diff.astype(int)
    print('acc= ', np.mean(np.sqrt(np.square(images_current_landmarks - true_landmarks))))
  return images_current_landmarks, regressors, pcas