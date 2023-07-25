""" Research Project: Cancer Image Classification

This project trains a model to perform two tasks:
1. detect the organ for a medical image
2. classify it into cancer or benign

The project uses the Multi Cancer Dataset from Kaggle: 
https://www.kaggle.com/datasets/obulisainaren/multi-cancer


"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.utils import img_to_array, array_to_img, save_img
import matplotlib.pyplot as plt
import pandas as pd
from random import randrange

batch_size=16
organ_labels = ["Blood", "Brain", "Breast", "Cervix", "Colon", "Kidney", "Lung", "Lymph", "Mouth"]
cancer_labels = ["Benign", "Malignant"]

def load_image_dataset(data_dir):
  """Loads an image dataset from a folder, where each class is in a separate subfolder."""
  dataset = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      labels='inferred',
      label_mode='int',
      image_size=(256,256),
      shuffle=True,
      batch_size=batch_size)
  return dataset

def split_test_dataset(dataset, train_fraction=0.8, val_fraction=0.15, test_fraction=0.05):
  """Splits a dataset into training, evaluation, and test sets."""
  num_samples = len(dataset)
  train_size = int(num_samples * train_fraction)
  val_size = int(num_samples * val_fraction)
  test_size = num_samples - train_size - val_size
  test_dataset = dataset.skip(train_size + val_size)
  return test_dataset

def load_test_dataset(path):
  dataset = load_image_dataset(path)
  num_classes = len(dataset.class_names)
  test_dataset = split_test_dataset(dataset)
  return test_dataset

def test_classification(model_path, dataset_path, label_names):
  # Load model for prediction  
  model = keras.models.load_model(model_path)  
  print(model.summary())
  
  # Load test dataset
  ds = load_test_dataset(dataset_path)

  print("Finished loading test dataset: " + str(len(ds)))
  
  rand = randrange(5,len(ds))
  images, labels = next(iter(ds.skip(rand).take(1)))

  i = 1
  for img,lbl in zip(images, labels):
    print("Predicting for image " + str(i))
    #plt.imshow(array_to_img(img))
    #plt.show()
    save_img('Results/test_' + dataset_path + '_' + str(i) + '.png', img)
    x = np.expand_dims(img, axis=0)
    score = model.predict(x).flatten()
    index = score.argmax(axis=0)    
    print('Predicted:', label_names[index], " and correct label is: ", label_names[lbl.numpy().item()])
    i+=1



if __name__ == "__main__":
  test_classification("rp_train_organ_202307251441.h5", "Organ", organ_labels)
  test_classification("rp_train_cancer_202307251452.h5", "Cancer", cancer_labels)

