#! /usr/bin/env python
import cv2
import numpy as np
from keras.models import load_model

from config import infer as config
from src.utils.utils import decode_netout, draw_boxes
from src.networks.feature_extractors import get_feature_extractor


class Prediction():
    
    def __init__(self):        
        model_path = config["model_path"]
        self.backend = config['backend']
        self.input_size = config["input_size"]
        self.max_box_per_image = config["max_box_per_image"]
        self.anchors = config["anchors"]
        self.labels = config["labels"]        
        self.nb_class = len(self.labels)
        
        self.model = load_model(model_path, compile=False)
        feature_extractor, grid_h, grid_w = get_feature_extractor(self.backend, self.input_size)
        self.normalize = feature_extractor.normalize
        
    def predict(self, image):

#        image = cv2.imread(image_path)
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.normalize(image)
    
        input_image = image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = np.zeros((1,1,1,1,self.max_box_per_image,4))
    
        netout = self.model.predict([input_image, dummy_array])[0]
        boxes  = decode_netout(netout, self.anchors, self.nb_class)
        
        image = draw_boxes(image, boxes, self.labels)
                
        return image, boxes

