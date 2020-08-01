#importing other required libraries
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))

train_dir = "./Train"
test_dir = "./Test"
validation_dir ="./Validation"
"""Extracting features using the pretrained convolutional base"""
datagen =ImageDataGenerator(rescale=1./255)
batch_size = 20
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count,8))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels
train_features, train_labels = extract_features(train_dir, 5000)
validation_features, validation_labels = extract_features(validation_dir, 364)
#test_features, test_labels = extract_features(test_dir, 156)
train_features = np.reshape(train_features, (5000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (364, 4 * 4 * 512))
#test_features = np.reshape(test_features, (156, 4 * 4 * 512))
conv_base.summary()
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])
history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))


model.save('DanceClassification.h5')
#joblib.dump(model, "DanceClassification.h5")
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

from keras.preprocessing import image
target_size=(150, 150, 3)
img = image.load_img("./preview/bharatanatyam_0_4.jpeg", target_size=target_size)
img = np.expand_dims(img, axis=0)
img = img/255.0
feature_value = conv_base.predict(img)
feature_value= np.reshape(feature_value,(1,4*4*512))
y_classes = model.predict_classes(feature_value)
if y_classes == 0:
    print('Bharatanatyam')
elif y_classes==1:
    print('kathak')
elif y_classes==2:
    print('kathakali')
elif y_classes==3:
    print('kuchipudi')
elif y_classes==4:
    print('Manipuri')
elif y_classes==5:
    print('mohiniyattam')
elif y_classes==6:
    print('odissi')
else:
    print('sattriya')


