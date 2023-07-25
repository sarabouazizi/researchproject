# Research Project by Sara Bouazizi
# Classification of medical images into organ and detection of cancer

In this project, I use the Multi Cancer image database from [Kaggle][https://www.kaggle.com/datasets/obulisainaren/multi-cancer).
There are two tasks that are trained. The first one classifies the image into the organ that is in the image. There are 9 classes: 
```python
organ_labels = ["Blood", "Brain", "Breast", "Cervix", "Colon", "Kidney", "Lung", "Lymph", "Mouth"]
```

The second task is trained to detect cancer in the image. There are 2 classes:
```python
cancer_labels = ["Benign", "Malignant"]
```
