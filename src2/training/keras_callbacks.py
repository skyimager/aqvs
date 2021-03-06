#Standard imports
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

#Custom imports
from config import objd as config


class GPUModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, org_model, **kwargs):
        """
        :param filepath:
        :param org_model: Keras model to save instead of the default.
            This is used especially when training multi-gpu models built with Keras multi_gpu_model().
            In that case, you would pass the original "template model" to be saved each checkpoint.
        :param kwargs: Passed to ModelCheckpoint.
        """

        self.org_model = org_model
        super().__init__(filepath, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        model_before = self.model
        self.model = self.org_model
        super().on_epoch_end(epoch, logs)
        self.model = model_before


def get_callbacks(model):

    #Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=config["train"]["min_delta"],
                                   patience=config["train"]["patience_es"], verbose=1,
                                   mode = "min")
    #LR manage
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=config["train"]["factor_lr"],
                                  patience=config["train"]["patience_lr"], min_lr=1e-8,
                                  min_delta=config["train"]["min_delta"], verbose=1)

    #TB visualzation
    tensorb = TensorBoard(log_dir="./data/"+config["exp_name"]+"/"+"Graph", histogram_freq=0,
                          write_graph=True, write_images=True)

    #Model Checkpoint
    if config["train"]["no_of_gpu"] > 1:
        filepath= "./data/"+config["exp_name"]+"/"+"gpu_checkpoint-{epoch:02d}-{val_loss:.2f}.h5"
        checkpoint = GPUModelCheckpoint(filepath, model, monitor='val_loss', mode="min",
                                 save_best_only=False, save_weights_only=False,
                                 verbose=1)

    else:
        filepath= "./data/"+config["exp_name"]+"/"+"checkpoint-{epoch:02d}-{val_loss:.2f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', mode="min",
                                 save_best_only=False, save_weights_only=False,
                                 verbose=1)

    callbacks_list = [checkpoint, reduce_lr, early_stopping, tensorb]

    return callbacks_list
