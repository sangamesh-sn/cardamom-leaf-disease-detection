import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

print("TensorFlow version:", tf.__version__)

# Paths to image folders
train_folder = "ddd_ds/train"
val_folder = "ddd_ds/validation"
test_folder = "ddd_ds/test"

# Number of classes in the dataset
num_classes = 2

# Input image dimensions
input_size = (224, 224)

# Training parameters
batch_size = 512
epochs = 5

# Image preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load validation data
validation_generator = val_test_datagen.flow_from_directory(
    val_folder,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Load test data
test_generator = val_test_datagen.flow_from_directory(
    test_folder,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load pre-trained ResNet152V2 model
base_model = ResNet152V2(
    include_top=False,
    weights='imagenet',
    input_shape=(input_size[0], input_size[1], 3),
    pooling='avg',
    classifier_activation=None
)

# Add custom layers
x = base_model.output
x = Dense(1024, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=output)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Show model structure
model.summary()

# Train model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Evaluate model on test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save training history to DataFrame
history_df = pd.DataFrame(history.history)
history_df['test_loss'] = test_loss
history_df['test_accuracy'] = test_accuracy

# -------------------- 📊 Plot Training Curves --------------------
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

# -------------------- 📑 Predictions & Reports --------------------
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Class labels
class_labels = list(test_generator.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(test_generator.classes, predicted_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png")
plt.show()

# Classification Report
report = classification_report(test_generator.classes, predicted_classes,
                               target_names=class_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("Classification Report:")
print(report_df)

# -------------------- 📄 Save Test Results --------------------
test_results = []
for i, image_path in enumerate(test_generator.filepaths):
    image_name = os.path.basename(image_path)
    true_class = class_labels[test_generator.classes[i]]
    predicted_class = class_labels[predicted_classes[i]]
    accuracy = 100 if true_class == predicted_class else 0
    test_results.append({
        "Image Name": image_name,
        "Predicted Class": predicted_class,
        "True Class": true_class,
        "Correct": "Yes" if accuracy == 100 else "No"
    })

test_results_df = pd.DataFrame(test_results)

# -------------------- 📘 Save Everything in ONE Excel --------------------
excel_filename = f"ResNet152V2_{num_classes}classes_results.xlsx"
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    history_df.to_excel(writer, sheet_name="Training_History", index=False)
    report_df.to_excel(writer, sheet_name="Classification_Report")
    test_results_df.to_excel(writer, sheet_name="Test_Results", index=False)

print(f"✅ All results saved in one Excel file: {excel_filename}")

# -------------------- 💾 Save TensorFlow Lite Model --------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_model_filename = f"model_ResNet152V2_{num_classes}classes.tflite"
with open(tflite_model_filename, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Model saved in TensorFlow Lite format: {tflite_model_filename}")
