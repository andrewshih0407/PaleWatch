import tensorflow as tf
import tensorflowjs as tfjs
import os, json

tf_path   = "/kaggle/working/palewatch_tf"
tfjs_path = "/kaggle/working/palewatch_tfjs"
os.makedirs(tfjs_path, exist_ok=True)

print("Loading SavedModel...")
model = tf.saved_model.load(tf_path)

print("Converting to TFJS...")
tfjs.converters.convert_tf_saved_model(
    tf_path,
    tfjs_path,
)

with open(f"{tfjs_path}/class_map.json", "w") as f:
    json.dump({"bleached": 0, "healthy": 1}, f)

print("\nFiles in palewatch_tfjs/:")
for fname in sorted(os.listdir(tfjs_path)):
    size = os.path.getsize(os.path.join(tfjs_path, fname))
    print(f"  {fname:45s} {size/1024:6.0f} KB")
print("\nDone — download palewatch_tfjs/ from the output panel")
