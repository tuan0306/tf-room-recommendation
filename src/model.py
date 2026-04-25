import tensorflow as tf
layers=tf.keras.layers
 
def build_room_model(vectorize_layer, price_norm, area_norm,vocab_size=1000,embedding_dim=16):
    input_desc=layers.Input(shape=(1,),dtype="string",name="description")
    x1=vectorize_layer(input_desc)
    x1=layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim)(x1)
    x1=layers.GlobalAveragePooling1D()(x1)
    
    input_price=layers.Input(shape=(1,),name="price")
    x2=price_norm(input_price)
    
    input_area=layers.Input(shape=(1,),name="area")
    x3=area_norm(input_area)
    
    combined=layers.Concatenate()([x1,x2,x3])
    
    z=layers.Dense(units=16,activation="relu")(combined)
    z=layers.Dense(units=8,activation="relu")(z)
    output=layers.Dense(units=1,activation="sigmoid")(z)
    
    model=tf.keras.Model(
        inputs=[input_desc,input_price,input_area],
        outputs=output
    )
    
    return model