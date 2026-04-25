import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import TextVectorization, Normalization

def load_room_dataset(csv_path,batch_size=32, max_tokens=1000, output_sequence_length=15):
    
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    vectorize_layer=TextVectorization(
        max_tokens=max_tokens,
        output_sequence_length=output_sequence_length,
        output_mode='int'
    )
    vectorize_layer.adapt(df["Description"].values)
    
    price_norm=Normalization(axis=None)
    price_norm.adapt(df["Price"].values)
    
    area_norm=Normalization(axis=None)
    area_norm.adapt(df["Area"].values)
    
    
    
    X={
        "description":df["Description"].values,
        "price":df["Price"].values,
        "area":df["Area"].values
    }
    
    y=df.pop("Label")
    
    ds=tf.data.Dataset.from_tensor_slices((X,y))
    ds=(
        ds
        .shuffle(buffer_size=len(df))
        .batch(batch_size=batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds, vectorize_layer, price_norm,area_norm