import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn.metrics
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import multilabel_confusion_matrix
import tensorflow as tf


#Read the given the .mat file using read_file function
def read_file(path):
    #path = r'C:\Users\manoj_9hatybv\OneDrive\Desktop\fisheriris_matlab.mat'
    mat = loadmat(path)
    input_data = mat['meas'] #input_data.shape  = 150x4
    input_data = input_data
    output_data = mat['species'] #output_data.shape = 150x1
    d=pd.DataFrame(input_data)
    d.insert(4,'color',output_data)
    return d

#Dividing the data set into train,validate,test dataset
def BT19ECE064_dataset_div_shuffle(file_path,ratio):
    data = read_file(file_path) # read_file function called 
    r = (1+ratio)/2
    train, validate, test = np.split(data.sample(frac=1, random_state=42),[int(ratio*len(data)), int(r*len(data))])
    return train,validate,test

path = r'C:\Users\manoj_9hatybv\OneDrive\Desktop\fisheriris_matlab.mat' # give the path to dataset
ratio = 0.7
train_data,validate_data,test_data = BT19ECE064_dataset_div_shuffle(path,ratio)
# Declaring X_train,Y_train
X_train = train_data.iloc[:,:4].values # x_train data
Y_train = train_data.loc[:,'color']# Y_train data
Y_train = Y_train.astype('str')

###################################
 #Declaring X_test,Y_test
X_test = test_data.iloc[:,:4] #X_test data
Y_test = test_data.loc[:,'color'] # Y_test data
Y_test = Y_test.astype('str')

# Intializing the SVM and fitting the data
linear_SVM = SVC(kernel='linear', random_state = 1) # implementing SVM linear kernel
linear_SVM.fit(X_train,Y_train)

# Predicting the classes for test data for linear SVM
Y_pred = linear_SVM.predict(X_test)

# Attaching the predictions to test set for comparing
test_data["Predictions_linear"] = Y_pred

# implementing SVM polynomial kernel with degree 10
poly_SVM = SVC(kernel='poly', degree = 10) 
poly_SVM.fit(X_train,Y_train)

Y_pred2 = poly_SVM.predict(X_test)
test_data["Predict_poly"] = Y_pred2

# implementing SVM rbf or gaussian kernel
rbf_SVM = SVC(kernel='rbf', random_state = 42) 
rbf_SVM.fit(X_train,Y_train)
Y_pred3 = rbf_SVM.predict(X_test)
test_data["Predict_Gaussian"] = Y_pred2

# definition of grid_search which performs grid search to  fine tune hyperparameters
def grid_search(train_x,train_y):
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf','poly','linear']} 
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
    grid_pred = grid.fit(X_train, Y_train)
    best_para = grid.best_params_
    #print('\n\n')
    best_est = grid.best_estimator_
    return grid_pred,best_para,best_est


# performing the grid search to fine tune hyperparameters
Grid_Pred,para,est = grid_search(X_train,Y_train)

# plotting the accuracy vs hyperparameters plot  
q = Grid_Pred.cv_results_['mean_test_score']
p = Grid_Pred.cv_results_['param_C']
#p = Grid_Pred.cv_results_['param_kernel']
x_p = np.array(p)
y_p = np.array(q)
#plt.scatter(x_p,y_p)
plt.plot(x_p,y_p)
plt.show()

#results function displays the accuracy,sensitivity,specificity ,ROC, AUROC of the model.
def results(true_y,pred_y): 
    #cm = confusion_matrix(test_y,pred_y)
    cm = multilabel_confusion_matrix(true_y, pred_y, labels=["SETOSA", "VERSICOLOR", "VIRGINICA"])
    accuracy = accuracy_score(true_y,pred_y)
    #cm_df = pd.DataFrame(cm,
                     #index = ['SETOSA','VERSICOLOR','VIRGINICA'], 
                     #columns = ['SETOSA','VERSICOLOR','VIRGINICA'])
    colors = ['SETOSA','VERSICOLOR','VIRGINICA']
    class_report = sklearn.metrics.classification_report(true_y, pred_y, target_names= colors)
    #sensitivity or recall = 
    #specificity = 
    #ROC   =
    #AUROC =
    print("The confusion matrix is : \n\n",cm)
    print(class_report)
    print("\n Accuracy for chosen SVM kernel is : ",accuracy)

results(Y_test,Y_pred)

# creating ANN model to solve the same problem
ann = tf.keras.models.Sequential()
#Adding First Hidden Layer
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
#Adding Second Hidden Layer
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
#Adding Output Layer
ann.add(tf.keras.layers.Dense(units=3,activation="softmax"))
#Compiling ANN
ann.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])
#Fitting ANN
ann.fit(X_train,Y_train,batch_size=32,epochs = 100)
