objd = {
    "exp_name": "yolo_t1",

    "model" : {
        "backend":              "Tiny Yolo",
        "input_size":           300,
        "anchors":              [0.35,0.49, 0.44,1.01, 0.46,2.35, 0.88,1.16, 1.09,0.45, 1.49,0.80, 1.57,1.72],
        "max_box_per_image":    3,
        "labels":               ["crack", "wrinkle"]
    },

    "train": {
        "train_image_folder":   "data/images/",
        "train_annot_folder":   "data/annotations/",

        "no_of_gpu":            1,
        "batch_size":           1*16,

        "train_times":          8,
        "pretrained_weights":   "",
        "learning_rate":        1e-4,
        "nb_epochs":            15,
        "warmup_epochs":        3,

        "patience_lr":          5,
        "min_delta":            0.001,
        "factor_lr":            0.5,
        "patience_es":          30,

        "object_scale":         5.0 ,
        "no_object_scale":      1.0,
        "coord_scale":          1.0,
        "class_scale":          1.0
    }
}

infer = {
    "backend":              "Tiny Yolo",
    "input_size":           300,
    "anchors":              [0.35,0.49, 0.44,1.01, 0.46,2.35, 0.88,1.16, 1.09,0.45, 1.49,0.80, 1.57,1.72],
    "max_box_per_image":    3,
    "labels":               ["crack", "wrinkle"],
    "model_path":           "data/yolo_t1/"
}

retinanet = {
    'annotations': "/data/aqvs/data/retinanet/train_annotations.csv",   #help='Path to CSV file containing annotations for training.'
    'classes': "/data/aqvs/data/retinanet/class_mapping.csv",           #help='Path to a CSV file containing class label mapping.'
    'val-annotations':"/data/aqvs/data/retinanet/val_annotations.csv",  #help='Path to CSV file containing annotations for validation.'

    'resume-training':False,
    'snapshot':"",                                                      #help='Resume training from a snapshot.'
    'imagenet-weights':True,                                            #help='Initialize the model with pretrained imagenet weights.
    'weights':"/data/aqvs/data/pretrained/resnet50_coco_best_v2.0.1.h5",#help='Initialize the model with weights from a file.'

    'backbone':"resnet50",                                              #help='Backbone model used by retinanet.'
    'batch-size':16,                                                    #help='Size of the batches.'
    'multi-gpu':1,                                                      #help='Number of GPUs to use for parallel processing.'
    'multi-gpu-force': False,                                           #help='Extra flag needed to enable (experimental) multi-gpu support.'
    'epochs':10,                                                        #help='Number of epochs to train.'
    'steps':100,                                                        #help='Number of steps per epoch.'
    'lr':0.001,                                                         #help='Learning rate.'
    'snapshot-path':"./snapshots",                                      #help='Path to store snapshots of models during training
    'tensorboard-dir':"./logs",                                         #help='Log directory for Tensorboard output'

    'freeze-backbone':"",                                               #help='Freeze training of backbone layers.'
    'random-transform':True,
    'image-min-side':800,                                               #help='Rescale the image so the smallest side is min_side.'
    'image-max-side':1200,                                              #help='Rescale the image if the largest side is larger than max_side.'

    'weighted-average': True,                                           #help='Compute the mAP using the weighted average of precisions among classes.'
    'compute-val-loss': True,                                           #help='Compute validation loss during training', dest='compute_val_loss'
    'workers':3,                                                        #help='Number of multiprocessing workers. To disable multiprocessing, set workers to 0'
    'max-queue-size':10                                                 #help='Queue length for multiprocessing workers in fit generator.'
}
