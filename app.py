import matplotlib.pyplot as plt

import data_preprocessing
import CNN

if __name__ == "__main__":
    # display classes
    data_preprocessing.print_classes()
    
    #fetch train and test sets
    x_train,y_train,x_test,y_test = data_preprocessing.fetch_train_test_sets(print_info=True)
    
    # # ploting image from training set    
    # sample = 1999
    # plt.imshow(x_train[sample])
    # plt.axis('off')
    # plt.show
    
    #________________CNN________________
    
    # # printing summary
    # CNN.print_model_summary()
    
    # traing
    CNN.train_model(x_train, y_train,120)
    
    # evaluting
    CNN.evaluate_model(x_test, y_test)
    
    
    
    
    
    
    
    
    
    
    
    
    
    