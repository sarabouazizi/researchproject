# Research Project by Sara Bouazizi
# Classification of medical images into organ and detection of cancer

## Introduction

## Experiment

In this project, I use the Multi Cancer image database from [Kaggle][https://www.kaggle.com/datasets/obulisainaren/multi-cancer).
There are two tasks that are trained. The first one classifies the image into the organ that is in the image. There are 9 classes: 
```python
organ_labels = ["Blood", "Brain", "Breast", "Cervix", "Colon", "Kidney", "Lung", "Lymph", "Mouth"]
```

The second task is trained to detect cancer in the image. There are 2 classes:
```python
cancer_labels = ["Benign", "Malignant"]
```

A test script is provided to pick a random batch of images and then run both tasks and output the results.

Both models use the pre-trained InceptionV3 model from *tensorflow.keras.applications*.
The model top is replaced to do the required classification task. The model top code is:
```python
model_top = Sequential()
model_top.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:], data_format=None)),
model_top.add(Dense(256, activation='relu'))
model_top.add(Dropout(0.2))
model_top.add(Dense(num_classes))
```

The model summary for the organ classiciation is shown below:
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_2 (InputLayer)        [(None, 256, 256, 3)]     0

 rescaling (Rescaling)       (None, 256, 256, 3)       0

 tf.math.truediv (TFOpLambda  (None, 256, 256, 3)      0
 )

 tf.math.subtract (TFOpLambd  (None, 256, 256, 3)      0
 a)

 inception_v3 (Functional)   (None, 6, 6, 2048)        21802784

 sequential (Sequential)     (None, 9)                 526857

=================================================================
Total params: 22,329,641
Trainable params: 526,857
Non-trainable params: 21,802,784
_________________________________________________________________
```

The model summary for the cancer detection is shown below:
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_2 (InputLayer)        [(None, 256, 256, 3)]     0

 rescaling (Rescaling)       (None, 256, 256, 3)       0

 tf.math.truediv (TFOpLambda  (None, 256, 256, 3)      0
 )

 tf.math.subtract (TFOpLambd  (None, 256, 256, 3)      0
 a)

 inception_v3 (Functional)   (None, 6, 6, 2048)        21802784

 sequential (Sequential)     (None, 2)                 525058

=================================================================
Total params: 22,327,842
Trainable params: 525,058
Non-trainable params: 21,802,784
_________________________________________________________________
```

## Results

The model for detecting the organ in the image has the following training results:
>> 
When testing on random images, the following results are shown:
>> 

The model for predicting if the image shows cancer or not has the following training results:
>> 
When testing on random images, the following results are shown:
>> 

## Conclusion


