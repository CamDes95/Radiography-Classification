import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

os.chdir("/Users/CaM/Desktop/Kaggle/Radiography Classification/")


##### IMPORTATION DES DONNEES #####

# Trouver tous les chemins vers les fichiers images
liste = glob.glob('./Training/*/*.jpg')

# Extraire les labels de chaque image
liste = list(map(lambda x : [x, x.split('/')[2]], liste))

df = pd.DataFrame(liste, columns=['filepath', 'nameLabel'])
df['label'] = df['nameLabel'].replace(df.nameLabel.unique(), [*range(len(df.nameLabel.unique()))])
df.head()

# Afficher une image
filepath = df.filepath[0]
im = tf.io.read_file(filepath)
im = tf.image.decode_jpeg(im, channels=3)
tf.image.resize(im, size=(256,256))
plt.imshow(im);


##### SEPARATION DES DONNEES #####

X_train_path, X_test_path, y_train, y_test = train_test_split(df.filepath, df.label, train_size=0.8, random_state=456)

# Définition X_test
X_test = []

for filepath in tqdm(X_test_path):

    im = tf.io.read_file(filepath)
    im = tf.image.decode_jpeg(im, channels=3)
    im = tf.image.resize(im, size=(256,256))

    X_test.append([im])

X_test = tf.concat(X_test, axis=0)


# Définition du dataset_train 
@tf.function

def load_image(filepath, resize=(256,256)):
    im = tf.io.read_file(filepath)
    im = tf.image.decode_png(im, channels=3)
    return tf.image.resize(im, resize)


dataset_train = tf.data.Dataset.from_tensor_slices((X_train_path, y_train))

dataset_train = dataset_train.map(lambda x, y : [load_image(x), y], num_parallel_calls=-1).batch(32)




##### MODELISATION #####

# Chgt des couches d'extraction de feature d'efficientNet
from tensorflow.keras.applications import EfficientNetB1

efficientNet = EfficientNetB1(include_top=False, input_shape=(256,256,3)) # weights de BDD Imagenet

# Freeze backbone
for layer in efficientNet.layers:
    layer.trainable = False

efficientNet.summary()

# Ajout des couches pour classification
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.models import Model, Sequential, load_model

model = Sequential()

model.add(efficientNet)
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))

model.summary()

# Compilation du modèle
model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

# Définition des rappels

from tensorflow.keras import callbacks
# Sauvegarder automatiquement les poids
checkpoint = callbacks.ModelCheckpoint(filepath = 'checkpoint', 
                                       monitor = 'val_loss',
                                       save_best_only = True,
                                       save_weights_only = False,
                                       mode = 'min',
                                       save_freq = 'epoch')

# diminue automatiquement le LR
lr_plateau = callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                         patience=5,
                                         factor=0.1,
                                         verbose=2,
                                         mode='min')


##### ENTRAINEMENT #####

history = model.fit(dataset_train, epochs=10, validation_data=(X_test, y_test), callbacks = [lr_plateau, checkpoint])

# Sauvegarde du modèle
model.save("Model_Radio_Classifier.h5")

# Visualisation loss et dice_coef lors de l'entraînement :
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(history.history["loss"], label = "loss")
plt.plot(history.history["val_loss"], label = "val loss")
plt.xlabel("epochs")
plt.ylabel("loss function")
plt.legend()

plt.subplot(122)
plt.plot(history.history["accuracy"], label="accuracy", color="red")
plt.plot(history.history["val_accuracy"], label="val accuracy", color="green")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.show();


##### PREDICTION #####

# Probability prediction
y_prob = model.predict(X_test, batch_size=64)

# Label prediction
y_pred = tf.argmax(y_prob, axis=-1).numpy()


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print('Accuracy :', accuracy_score(y_test, y_pred),"\n")
print(confusion_matrix(y_test, y_pred))


##### VISUALISATION RESULTATS #####

indices_random = tf.random.uniform([4], 0, len(X_test), dtype=tf.int32)

plt.figure(figsize=(15,7))
for i, idx in enumerate(indices_random):
    plt.subplot(1,4,i+1)
    plt.imshow(tf.cast(X_test[idx], tf.int32))
    plt.xticks([])
    plt.yticks([])
    plt.title('Pred class : {} \n Real class : {}'.format(df.nameLabel.unique()[y_pred[idx]], df.nameLabel.unique()[y_test.values[idx]]))




