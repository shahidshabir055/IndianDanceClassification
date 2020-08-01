from keras.preprocessing import image
from keras.applications import VGG16
import numpy as np
from keras.models import load_model
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
target_size=(150, 150, 3)
img = image.load_img("./preview/14.jpg", target_size=target_size)
img = np.expand_dims(img, axis=0)
img = img/255.0
model = load_model('DanceClassification.h5')
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
