import tensorflow as tf
from keras import layers
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input
import os
import json

# =========================
# CONFIG
# =========================
DATA_DIR = "dataset/images"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8

MODEL_PATH = "weights/bird_model.keras"
CLASS_MAP_PATH = "weights/class_map.json"

os.makedirs("weights", exist_ok=True)


SELECTED_CLASSES = [
    "073.Blue_Jay",
    "076.Dark_eyed_Junco",
    "118.House_Sparrow",
    "132.White_crowned_Sparrow",
    "133.White_throated_Sparrow",
    "047.American_Goldfinch",
    "091.Mockingbird",
    "094.White_breasted_Nuthatch",
    "192.Downy_Woodpecker",
    "189.Red_bellied_Woodpecker"
]

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=SELECTED_CLASSES,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=SELECTED_CLASSES,
)

# =========================
# CLASS ORDER (0–9 automatically)
# =========================
class_names = train_ds.class_names
num_classes = len(class_names)

print("\nTraining on classes:")
for i, c in enumerate(class_names):
    print(i, c)

# =========================
# SAVE CLEAN LABEL MAP FOR STREAMLIT
# =========================
clean_labels = [c.split(".", 1)[1].replace("_", " ") for c in class_names]

class_map = {i: clean_labels[i] for i in range(num_classes)}

with open(CLASS_MAP_PATH, "w") as f:
    json.dump(class_map, f)

# =========================
# PERFORMANCE OPTIMIZATION
# =========================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# =========================
# DATA AUGMENTATION
# =========================
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),

    # geometric realism
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),

    # lighting variation (matches low light demo)
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),

    # sensor realism (matches noise demo)
    layers.GaussianNoise(0.05),
])

# =========================
# MODEL (EfficientNetB0)
# =========================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_aug(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# =========================
# TRAIN HEAD
# =========================
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# =========================
# FINE TUNING
# =========================
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
)

# =========================
# SAVE MODEL
# =========================
model.save(MODEL_PATH)

print("\nSaved model to:", MODEL_PATH)
print("Saved class map to:", CLASS_MAP_PATH)