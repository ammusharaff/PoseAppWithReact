import tensorflow as tf
import os

# Get the absolute path to the folder where this script (test.py) lives
# e.g., /home/dinesh/Documents/pose-web-app/backend
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the models directory
# e.g., /home/dinesh/Documents/pose-web-app/backend/src/models
saved_model_dir = os.path.join(BASE_DIR, 'src', 'models')
output_file = os.path.join(BASE_DIR, 'src', 'models', 'MoveNet_Thunder.tflite')

print(f"Looking for model in: {saved_model_dir}")

# Verify the path exists before running converter
if not os.path.exists(saved_model_dir):
    print(f"❌ Error: Directory not found: {saved_model_dir}")
    exit(1)

variable_index = os.path.join(saved_model_dir, 'variables', 'variables.index')
if not os.path.exists(variable_index):
    print(f"❌ Error: Variables not found at: {variable_index}")
    exit(1)

# Run Converter
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

try:
    tflite_model = converter.convert()
    with open(output_file, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ Success! Saved to: {output_file}")
except Exception as e:
    print(f"❌ Conversion failed: {e}")