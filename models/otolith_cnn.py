import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from PIL import Image
import numpy as np
from keras.preprocessing import image

def build_otolith_model(input_shape=(128,128,3), num_classes=5):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model



from PIL import Image
import numpy as np
import io

def predict_otolith(model, uploaded_file, labels):
    # Convert UploadedFile to BytesIO
    img_bytes = uploaded_file.read()        # read the bytes
    img_stream = io.BytesIO(img_bytes)      # convert to BytesIO

    # Open with PIL
    img = Image.open(img_stream)
    img = img.resize((128,128))
    
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Prediction
    pred_index = np.argmax(model.predict(img_array), axis=1)[0]
    return labels[pred_index]

