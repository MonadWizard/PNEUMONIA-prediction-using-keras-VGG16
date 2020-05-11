## Predicitng Models


from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
model = load_model('model_vgg19.h5')
img = image.load_img('val/NORMAL/IM-0003-0001.jpeg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
classes = model.predict(img_data)

# display accuracy
print('accuracy for normal: ', classes[0][0])

print('accuracy for Infected: ' ,classes[0][1])


