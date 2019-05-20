import os
import numpy as np
import fnmatch
import pickle

#preparing the data for train, val and test phases. If we do not need a seperate
#'test' phase, the same can be removed here. 
phases = ["train", "test", "validation"]

def test_train_split(base, datasets,config):
    """
    This script will split the dataset into train, test and validation and save
    the corresponding filespaths into a .txt files.
    As, an added feature it also saves the class_weights as a pickle file which
    can be later used during keras training to handle imbalanced dataset.
    """
    for dataset in datasets:

        if os.path.exists(os.path.join(base, dataset, "train.txt")):
            os.remove(os.path.join(base, dataset, "train.txt"))

        if os.path.exists(os.path.join(base, dataset, "validation.txt")):
            os.remove(os.path.join(base, dataset, "validation.txt"))

        if os.path.exists(os.path.join(base, dataset, "test.txt")):
            os.remove(os.path.join(base, dataset, "test.txt"))

        if os.path.exists(os.path.join(base, dataset, dataset+".pkl")):
            os.remove(os.path.join(base, dataset, dataset+".pkl"))

        files_per_klass = []
        no_of_files_per_klass = []
        class_weights = {}

        for klass in os.listdir(os.path.join(base, dataset)):
            if not klass.startswith("."):
                dire = os.path.join(base, dataset, klass)
                files = []

                #glob recursive doesn't work below 3.5
                for root, dirnames, filenames in os.walk(dire):
                    for filename in fnmatch.filter(filenames, '*.png'):
                        files.append(os.path.join(root, filename))

                files_per_klass.append((klass, files))
                no_of_files_per_klass.append(len(files))

        maxo = max(no_of_files_per_klass) #to be used for calculating class weights
        train_per = config[dataset]["train"]
        val_per = config[dataset]["validation"]

        for klass, files in files_per_klass:

            images = {}
            train_size = int(train_per*len(files))
            val_size = int(val_per*len(files))

            images["train"] =  files[:train_size]
            images["validation"] =  files[train_size:(train_size+val_size)]
            images["test"] =  files[(train_size+val_size):]

            #Generating integer label for the classes
            label = int(klass[-2:]) - 1

            #Generating class weights. The class with the highest number is given a
            #weight of 1.0.
            class_weights[label] = maxo//len(files)

            for phase in phases:
                with open('{}/{}.txt'.format(os.path.join(base,dataset),phase), 'a') as f:
                    for image in images[phase]:
                        image = os.path.relpath(image, os.path.join(base,dataset))
                        f.write('{} {}\n'.format(image, label))

        #Saving class weights as a pickle file
        with open('{}/{}.pkl'.format(os.path.join(base,dataset),dataset), 'wb') as f:
            pickle.dump(class_weights, f, pickle.HIGHEST_PROTOCOL)

        #Data shuffling
        for phase in phases:
            content = []
            with open('{}/{}.txt'.format(os.path.join(base,dataset),phase), 'r') as f:
                content = f.readlines()
            np.random.shuffle(content)
            print(dataset, " ", phase, " ", len(content))

            with open('{}/{}.txt'.format(os.path.join(base,dataset), phase), 'w') as f:
                for line in content:
                    f.write(line)

        print("file shuffling completed for {} \n".format(dataset))

if __name__ == "__main__":

    base = "data/English"

    #A single dataset can be composed of sub-datasets. The training can happen
    #either on the sub-dataset or complete dataset.
    datasets = ["GoodImg","BadImg"]

    #Train Val Test split per dataset
    config = {"GoodImg":{"train":0.70,
                    "validation":0.20,
                    "test":0.10},
          "BadImg":{"train":0.8,
                    "validation":0.10,
                    "test":0.10}
          }

    test_train_split(base, datasets,config)
