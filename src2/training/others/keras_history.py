
def generate_stats(history, config):

    history_to_save = history.history

    history_to_save['validation loss'] = min(history.history['val_loss'])
    history_to_save['validation accuracy'] = max(history.history['val_acc'])
    history_to_save['last validation loss'] = history.history['val_loss'][-1]
    history_to_save['last validation accuracy'] = history.history['val_acc'][-1]

    history_to_save['train accuracy'] = max(history.history['acc'])
    history_to_save['train loss'] = min(history.history['loss'])
    history_to_save['last train accuracy'] = history.history['acc'][-1]
    history_to_save['last train loss'] = history.history['loss'][-1]


    history_to_save['final lr'] = history.history['lr'][-1]
    history_to_save['total epochs'] = len(history.history['lr'])
    
    history_to_save['optimiser'] = config.optimiser 
    history_to_save['loss'] = config.loss
    history_to_save['metric'] = config.metric

    return history_to_save
