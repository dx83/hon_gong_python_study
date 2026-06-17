import pickle
from pathlib import Path

import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# =====================================================
# 설정
# =====================================================

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32

ROOT = Path.cwd()

CSV_PATH = ROOT / "pill_info_list.csv"
IMAGE_DIR = ROOT / "_images_crop_HSV"

MODEL_PATH = ROOT / "pill_shape_efficientnetb0.keras"
ENCODER_PATH = ROOT / "pill_shape_encoder.pkl"

AUTOTUNE = tf.data.AUTOTUNE

tf.random.set_seed(SEED)

# =====================================================
# CSV 로드
# =====================================================

print("CSV Loading...")

df_info = pd.read_csv(
    CSV_PATH,
    encoding="utf-8"
)

df_info["품목일련번호"] = (
    df_info["품목일련번호"]
    .astype(str)
)

df_info["의약품제형"] = (
    df_info["의약품제형"]
    .astype(str)
)

print("품목 수 :", len(df_info))

# =====================================================
# 품목일련번호 -> 제형
# =====================================================

shape_map = dict(
    zip(
        df_info["품목일련번호"],
        df_info["의약품제형"]
    )
)

# =====================================================
# 이미지 스캔
# =====================================================

print("Scanning Images...")

rows = []

for img_path in IMAGE_DIR.glob("*.png"):

    item_seq = img_path.name.split("_")[0]

    if item_seq not in shape_map:
        continue

    rows.append([
        str(img_path),
        item_seq,
        shape_map[item_seq]
    ])

df = pd.DataFrame(
    rows,
    columns=[
        "image_path",
        "item_seq",
        "shape"
    ]
)

print("이미지 수 :", len(df))

# =====================================================
# 결측 제거
# =====================================================

df = df.dropna()

df = df[
    df["shape"].str.len() > 0
]

print("사용 이미지 :", len(df))

# =====================================================
# Label Encoding
# =====================================================

encoder = LabelEncoder()

df["label"] = encoder.fit_transform(
    df["shape"]
)

NUM_CLASSES = len(
    encoder.classes_
)

print("클래스 수 :", NUM_CLASSES)

with open(
    ENCODER_PATH,
    "wb"
) as f:

    pickle.dump(
        encoder,
        f
    )

# =====================================================
# Group Split
# =====================================================

print("Creating Group Split...")

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

train_idx, valid_idx = next(
    sgkf.split(
        df,
        y=df["label"],
        groups=df["item_seq"]
    )
)

train_df = df.iloc[train_idx].reset_index(drop=True)
valid_df = df.iloc[valid_idx].reset_index(drop=True)

print()
print("Train :", len(train_df))
print("Valid :", len(valid_df))

train_groups = set(train_df["item_seq"])
valid_groups = set(valid_df["item_seq"])

overlap = len(
    train_groups.intersection(
        valid_groups
    )
)

print("Group Overlap :", overlap)

# =====================================================
# 전처리
# =====================================================

from tensorflow.keras.applications.efficientnet import (
    preprocess_input
)

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

# =====================================================
# 이미지 로드
# =====================================================

def load_image(path, label):

    img = tf.io.read_file(path)

    img = tf.image.decode_png(
        img,
        channels=3
    )

    img = tf.image.resize(
        img,
        (
            IMG_SIZE,
            IMG_SIZE
        )
    )

    img = preprocess_input(img)

    return img, label

# =====================================================
# Dataset
# =====================================================

def make_dataset(
    dataframe,
    training=False
):

    ds = tf.data.Dataset.from_tensor_slices(
        (
            dataframe["image_path"].values,
            dataframe["label"].values
        )
    )

    ds = ds.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    if training:

        ds = ds.map(
            lambda x, y: (
                augment(x),
                y
            ),
            num_parallel_calls=AUTOTUNE
        )

        ds = ds.shuffle(
            10000,
            seed=SEED
        )

    ds = ds.batch(
        BATCH_SIZE
    )

    ds = ds.prefetch(
        AUTOTUNE
    )

    return ds

# =====================================================
# Dataset 생성
# =====================================================

train_ds = make_dataset(
    train_df,
    training=True
)

valid_ds = make_dataset(
    valid_df,
    training=False
)

# =====================================================
# EfficientNetB0
# =====================================================

print("Building Model...")

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)

base_model.trainable = False

inputs = layers.Input(
    shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)

x = base_model(
    inputs,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = Model(
    inputs,
    outputs
)

# =====================================================
# 1차 학습
# =====================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [

    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    ),

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

print()
print("Stage 1 Training")

history1 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=15,
    callbacks=callbacks
)

# =====================================================
# Fine Tuning
# =====================================================

print()
print("Stage 2 Fine Tuning")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10,
    callbacks=callbacks
)

# =====================================================
# 최종 평가
# =====================================================

print()
print("Final Evaluation")

loss, acc = model.evaluate(
    valid_ds,
    verbose=1
)

print()
print(f"Validation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {acc:.4f}")

# =====================================================
# 저장
# =====================================================

model.save(
    MODEL_PATH
)

print()
print("Saved Model :", MODEL_PATH)
print("Saved Encoder :", ENCODER_PATH)