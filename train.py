import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Path to dataset
dataset_path = r"C:\Users\akhil\OneDrive\Documents\Resumes\parkinsons\Dataset"
categories = ["Healthy", "Parkinson"]
img_size = 128

data = []
labels = []

print("Loading dataset...")

for category in categories:
    folder_path = os.path.join(dataset_path, category)
    label = categories.index(category)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Equalize + resize
            img = cv2.equalizeHist(img)
            img = cv2.resize(img, (img_size, img_size))

            # Convert to RGB for MobileNetV2
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            data.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {img_name}: {e}")

# Convert to arrays & preprocess
data = np.array(data, dtype="float32")
data = preprocess_input(data)  # MobileNetV2 preprocessing
labels = to_categorical(labels, num_classes=2)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\nTraining samples: {len(x_train)}, Testing samples: {len(x_test)}")

# Load base model (MobileNetV2)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))

# Freeze first 100 layers (keep high-level features trainable)
for layer in base_model.layers[:100]:
    layer.trainable = False
for layer in base_model.layers[100:]:
    layer.trainable = True

# Custom head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile model
optimizer = Adam(learning_rate=1e-4)  # Lower LR for fine-tuning
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Stronger augmentation
datagen = ImageDataGenerator(
    rotation_range=25,
    zoom_range=0.25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
datagen.fit(x_train)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ModelCheckpoint("best_parkinson_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1)
]

# Train the model
print("\nTraining MobileNetV2 (fine-tuned) model...")
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=16),
    validation_data=(x_test, y_test),
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"\n Final Test Accuracy: {accuracy * 100:.2f}%")

# Save final model
model.save("parkinson_detector.h5")
print(" Model saved as parkinson_detector.h5")
