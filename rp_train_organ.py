""" Research Project: Cancer Image Classification

This project trains a model to perform two tasks:
1. detect the organ for a medical image
2. classify it into cancer or benign

The project uses the Multi Cancer Dataset from Kaggle: 
https://www.kaggle.com/datasets/obulisainaren/multi-cancer


"""

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os

"""Dataset loading

"""

batch_size=16
epochs = 10
organ_labels = ["Blood", "Brain", "Breast", "Cervix", "Colon", "Kidney", "Lung", "Lymph", "Mouth"]

def load_image_dataset(data_dir):
  """Loads an image dataset from a folder, where each class is in a separate subfolder."""
  dataset = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      labels='inferred',
      class_names=organ_labels,
      label_mode='int',
      image_size=(256,256),      
      shuffle=True,
      seed=123,
      batch_size=batch_size)
  return dataset

def split_dataset(dataset, train_fraction=0.8, val_fraction=0.15, test_fraction=0.05):
  """Splits a dataset into training, evaluation, and test sets."""
  num_samples = len(dataset)
  train_size = int(num_samples * train_fraction)
  val_size = int(num_samples * val_fraction)
  test_size = num_samples - train_size - val_size
  train_dataset = dataset.take(train_size)
  val_dataset = dataset.skip(train_size).take(val_size)
  test_dataset = dataset.skip(train_size + val_size)
  return train_dataset, val_dataset, test_dataset


dataset = load_image_dataset('Organ/')
num_classes = len(dataset.class_names)

train_dataset, val_dataset, test_dataset = split_dataset(dataset)


print('Training set size:', len(train_dataset))
print('Validation set size:', len(val_dataset))
print('Test set size:', len(test_dataset))

"""Next, initialize some variables that will be necessary for preparing and training the data."""

# SET VARIABLES img_width AND img_height equal TO 299
img_width, img_height = 256, 256

# SET VARIABLES train_samples AS 65, validation_samples AS 10,
nb_train_samples = len(train_dataset)
nb_validation_samples = len(val_dataset)


"""Next, we want to program the neural network architecture that will train the model. This has already been for you using the Inception V3 deep learning model."""

base_model = applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
#base_model.trainable = False
for layer in base_model.layers: layer.trainable = False
preprocess_input = tf.keras.applications.inception_v3.preprocess_input

model_top = Sequential()
model_top.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:], data_format=None)),
model_top.add(Dense(256, activation='relu'))
model_top.add(Dropout(0.2))
model_top.add(Dense(num_classes))

inputs = tf.keras.Input(shape=(img_width, img_height, 3))
#x = data_augmentation(inputs)
x = tf.keras.layers.Rescaling(1./255)(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
outputs = model_top(x)

model = Model(inputs=inputs, outputs=outputs)
print(model.summary())
#model = Model(inputs=base_model.input, outputs=model_top(base_model.output))

model.compile(optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

"""Now that we have our architecture, we are finally ready to train our model."""

# USE model.fit WHICH WILL INCLUDE THE ARGUMENTS:
# [TRAINING DATA], steps_per_epoch, epochs, validation_data, validation_steps
# https://www.rdocumentation.org/packages/keras/versions/2.11.1/topics/fit
with tf.device('/gpu:0'):
  history = model.fit(
                train_dataset,
                batch_size=batch_size,
                steps_per_epoch=600,
                epochs=epochs,                
                validation_data=val_dataset,
                validation_steps=60)

  # Save model    
  model.save('rp_train_organ_{}.h5'.format(datetime.now().strftime('%Y%m%d%H%M')))


  # Save history
  # USE plt.plot() TO PLOT:
  # TRAINING ACCURACY, TRAINING LOSS, VALIDATION ACCURACY, VALIDATION LOSS
  # https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
  plt.figure()
  plt.plot(history.history['accuracy'], 'orange', label='Training accuracy')
  plt.plot(history.history['val_accuracy'], 'blue', label='Validation accuracy')
  plt.plot(history.history['loss'], 'red', label='Training loss')
  plt.plot(history.history['val_loss'], 'green', label='Validation loss')
  plt.legend()

  RESULT_DIR = 'Results'
  save_name = 'rp_organ_history.png'
  plt.savefig(os.path.join(RESULT_DIR,save_name))

  hist_df = pd.DataFrame(history.history)
  hist_csv_file = os.path.join(RESULT_DIR,'rp_organ_history.csv')
  with open(hist_csv_file, mode='w') as f:
      hist_df.to_csv(f)