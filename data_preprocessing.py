import os
import numpy as np
import cv2 as cv
from sklearn.preprocessing import LabelEncoder


train_path = "./train_set/"
test_path = "./test_set/"

classes = os.listdir(train_path)

def print_classes(): print("classes",classes,sep=" : ")
    
def fetch_train_test_sets(print_info=False):
    print("preprocessing the dataset 🐱‍👤 ...")
    
    # installation
    x_train,y_train = [],[]
    x_test,y_test = [],[]
    
    # fetching the data
    try:
        for curr_class in classes:
            curr_class_train_set_title = os.listdir(train_path+curr_class)
            curr_class_test_set_title = os.listdir(test_path+curr_class)
            
            for sample_title in curr_class_train_set_title:
                sample = cv.imread(train_path+curr_class+"/"+sample_title)
                x_train.append(sample)
                y_train.append(curr_class)
                
            for sample_title in curr_class_test_set_title:
                sample = cv.imread(test_path+curr_class+"/"+sample_title)
                x_test.append(sample)
                y_test.append(curr_class)
    except Exception as e: 
        print(e)
            
    # converting the data into numpy arrays    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    # scaling the images
    x_train = x_train/255.0
    x_test = x_test/255.0
    
    # convert string fields into numeric fields
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)
    
    print("dataset is ready 🚀🚀 .")
    
    if(print_info):
        print("training set size",len(x_train),sep=" : ")
        print("testing set size",len(x_test),sep=" : ")
        
    return x_train,y_train,x_test,y_test
            
            
    
    
    
    
    
    
    