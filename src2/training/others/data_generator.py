"""
This is a utility class for Data Generation in Keras. To understand more about it
read the following blog post:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

"""

import keras
import numpy as np
import os
import cv2
import config

class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    """
    def __init__(self, base_path, file_paths, labels, preprocess, batch_size=32,
                 dim=(64,64), n_channels=3, n_classes=62, shuffle=True):

        self.base_path = base_path
        self.dim = dim
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.labels = labels
        self.file_paths = file_paths
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return: Number of batches per epoch
        """
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        """
        Returns one batch of data
        :param index: last batch number
        :return: features and labels
        """

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)

        return X, y

    def pre_process(self, image):

        processed = self.preprocess(image)

        return processed

    def on_epoch_end(self):
        """
        Post process hook at the end of the batch
        :return:
        """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """
        Returns training samples for the current batch
        :param indexes:
        :return: features and labels
        """

        X = []
        y = []

        for i, index in enumerate(indexes):

            # Store sample
            file_path = self.file_paths[index]
            label = self.labels[index]

            image = cv2.imread(os.path.join(self.base_path, file_path))
            if image is None:
                break
            # 'BGR'->'RGB'
            image = image[::-1, ...]
            resized = cv2.resize(image, self.dim,
                            interpolation = cv2.INTER_AREA)

            processed = self.pre_process(resized)

            X.append(processed)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        if config.loss == 'cat_cross':
            y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, y
