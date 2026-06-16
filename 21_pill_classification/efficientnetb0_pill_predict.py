import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 224

# ------------------------
# 모델 로드
# ------------------------

model = load_model("./pill_shape_efficientnetb0.keras")

with open("shape_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# ------------------------
# 이미지 전처리
# ------------------------

def predict_shape(image_path):

    img = tf.io.read_file(image_path)

    img = tf.image.decode_png(
        img,
        channels=3
    )

    img = tf.image.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    img = preprocess_input(img)

    img = tf.expand_dims(
        img,
        axis=0
    )

    # 예측
    pred = model.predict(
        img,
        verbose=0
    )

    class_idx = np.argmax(pred)

    class_name = encoder.inverse_transform(
        [class_idx]
    )[0]

    confidence = float(
        np.max(pred)
    )

    return class_name, confidence


shape, conf = predict_shape("./test1.png")

print("예측 제형 :", shape)
print("신뢰도 :", round(conf * 100, 2), "%")