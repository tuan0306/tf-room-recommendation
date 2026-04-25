import os
import sys
from pathlib import Path

if sys.flags.utf8_mode == 0:
    os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])

import tensorflow as tf

project_root = Path(__file__).resolve().parents[1]
model_path = project_root / 'models' / 'room_model_v1.keras'
model = tf.keras.models.load_model(model_path)

new_rooms = {
    "description": tf.constant(["Phòng trọ giá rẻ cho sinh viên, gần Làng Đại học, có máy lạnh"]),
    "price": tf.constant([3.2]), # 3.2 triệu
    "area": tf.constant([20.0])  # 20 m2
}

prediction = model.predict(new_rooms)
print(f"Xác suất sinh viên sẽ thích: {prediction[0][0] * 100:.2f}%")