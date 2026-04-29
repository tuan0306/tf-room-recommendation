TF Recommend Room
=================

Simple TensorFlow-based recommender for room suggestions.

Description
-----------
- Small project demonstrating dataset processing, model training, and prediction for recommending rooms.

Requirements
------------
- Python 3.8+
- TensorFlow

Quick Start
-----------
1. Prepare your data under the `data/` folder (`raw/` and `processed/`).
2. Train the model:
   - `python src/train.py`
3. Make predictions:
   - `python src/predict.py`

Project Layout
--------------
- `data/` — raw and processed datasets
- `models/` — saved model files
- `notebooks/` — exploratory notebooks
- `src/` — scripts: `dataset.py`, `model.py`, `train.py`, `predict.py`

