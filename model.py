import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, Flatten, Dropout, BatchNormalization, MaxPooling3D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import skimage.transform as skTrans
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage import transform as skTrans
from keras.utils import to_categorical
input_cancer = "C:\\Users\\junpa\\OneDrive\\Desktop\\Resized_Cancer"
input_control = "C:\\Users\\junpa\\OneDrive\\Desktop\\Resized_Control"
data = []
labels = []
for filename in os.listdir(input_cancer):
    full_path = os.path.join(input_cancer, filename)
    im = np.load(full_path)  # Load the NumPy array
    data.append(im)
    labels.append("cancer")
for filename in os.listdir(input_control):
    full_path = os.path.join(input_control, filename)
    im = np.load(full_path)  # Load the NumPy array
    data.append(im)
    labels.append("control")    
train_data = np.array(data)
train_labels = np.array(labels)
ohe_labels = np.zeros((len(train_labels), 2))
categories = np.array(["cancer", "control"])
for ii in range(len(train_labels)):
    jj = np.where(categories==labels[ii])
    ohe_labels[ii, jj] = 1
train_data, val_data, train_labels, val_labels = train_test_split(train_data, ohe_labels, test_size=0.3, random_state=42)
model = Sequential()
model.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu', input_shape=(400, 400, 15, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Flatten())
model.add(Dense(24, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))
opt = Adam(learning_rate=0.001)
early_stopping = EarlyStopping(patience = 10, restore_best_weights = True)
class_weights = {0: 1.0, 1: 1.0} 
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels, validation_split=0.3, epochs=100, batch_size=10, callbacks=[early_stopping], class_weight=class_weights)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
from sklearn.metrics import classification_report
y_pred = model.predict(val_data, batch_size=10)
y_pred_bool = np.argmax(y_pred, axis=1)
train_labels_int = np.argmax(val_labels, axis=1)
print(classification_report(train_labels_int, y_pred_bool))
from sklearn.metrics import confusion_matrix
result = confusion_matrix(train_labels_int, y_pred_bool, normalize='pred')
print(result)
