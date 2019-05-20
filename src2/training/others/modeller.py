import config
from keras.models import Model
from keras.layers import Dense, Dropout

def finetune_model(base_model):

    for i, layer in enumerate(base_model.layers):
        if i in config.freeze_layers:
            layer.trainable=False
        else:
            layer.trainable=True
    
    x = base_model.output
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    predictions = Dense(config.nclasses, activation='softmax')(x) # New softmax layer
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    print("Layer freezing complete!!")

    return finetune_model
