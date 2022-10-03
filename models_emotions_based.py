
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def data_prepation():
    label = preprocessing.LabelEncoder()
    blood=OrdinalEncoder()
    loyal=OrdinalEncoder()
    dataset= pd.read_csv("all_dat.csv",encoding="utf-8")
    dataset=dataset.drop(["Unnamed: 0"],axis=1)
    dataset_test= pd.read_csv("testing.csv",encoding="utf-8")
    dataset_test=dataset_test.drop(["Unnamed: 0"],axis=1)
    y=dataset.House
    y_test=dataset_test.House
    y_encod=label.fit_transform(y)
    y_encod_test=label.fit_transform(y_test)
    X=dataset.drop(["House","Skills","Loyalty"],axis=1)
    X_test=dataset_test.drop(["House","Skills","Loyalty"],axis=1)
    X["Blood status"] = X["Blood status"].replace(np.nan, "unknown")
    X_test["Blood status"] = X_test["Blood status"].replace(np.nan, "unknown")
    #X["Loyalty"] = X["Loyalty"].replace(np.nan, "unknown")
    #X_test["Loyalty"] = X_test["Loyalty"].replace(np.nan, "unknown")
    X['Male'] = X['Gender'].map( {'Male':1, 'Female':0} )
    X_test['Male'] = X_test['Gender'].map( {'Male':1, 'Female':0} )
    X["Blood status"]=blood.fit_transform(X[["Blood status"]])
    X_test["Blood status"]=blood.fit_transform(X_test[["Blood status"]])
    #X["Loyalty"]=blood.fit_transform(X[["Loyalty"]])
    #X_test["Loyalty"]=blood.fit_transform(X_test[["Loyalty"]])
    X=X.drop(["Character","Gender"],axis=1)
    X_test=X_test.drop(["Character","Gender"],axis=1)
    X.to_csv("final_encoding.csv")
    X_test.to_csv("final_encoding_test.csv")
    return (X,y,X_test,y_test)

def random_forest(X,y,X_test):
    clf = RandomForestClassifier(class_weight='balanced_subsample',criterion='gini',max_depth= 8,max_features= 'log2',n_estimators= 200)
    clf.fit(X,y)
    return clf.predict(X_test)

def knn(X,y,X_test):
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X,y)
    return  knn_classifier.predict(X_test)

def adaBoost(X,y,X_test):
    model = AdaBoostClassifier(n_estimators=50,base_estimator=DecisionTreeClassifier(max_depth= 10,max_features= 'auto'))
    model.fit(X,y)
    return  model.predict(X_test)

def prediction_results(y_test,predictions,methode_name):
    print("========="+methode_name+"=========")
    print("Score : " + str(accuracy_score(y_test, predictions)))
    list_of_lists = [list(y_test), list(predictions)]
    numpy_array = np.array(list_of_lists)
    transpose = numpy_array.T
    transpose_list = transpose.tolist()
    print(tabulate(transpose_list, headers=['labels', 'Predictions']))

data=data_prepation()
prediction_results(data[3],random_forest(data[0],data[1],data[2]),"Random Forest")
prediction_results(data[3],knn(data[0],data[1],data[2]),"KNeighborsClassifier")
#prediction_results(data[3],adaBoost(data[0],data[1],data[2]),"AdaBoostClassifier")