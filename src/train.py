import os
import sys
from pathlib import Path

if sys.flags.utf8_mode == 0:
    os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])

from dataset import load_room_dataset
from model import build_room_model

ds, vec_layer, p_norm, a_norm = load_room_dataset('data/raw/mock_data.csv')
model = build_room_model(
    vectorize_layer=vec_layer,
    price_norm=p_norm,
    area_norm=a_norm
)
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(ds,epochs=20)

keras_path = Path('models/room_model_v1.keras')
model.save(keras_path)
print(f"Saved Keras model to: {keras_path}")