from sklearn.model_selection import KFold,cross_val_score
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def data_load(path,sheet_name):
    data = pd.read_excel(path,sheet_name=sheet_name)
    X = data.values[: , 0:4]
    Y = data.values[: ,4]  
    Y = Y.astype('int')
    scaler = preprocessing.StandardScaler()
    x_minmax = scaler.fit_transform(X)
    data = [x_minmax,Y] 
    return data

def res_SVC(data,c_range,g_range):
    x_train,x_test,y_train,y_test = train_test_split(data[0],data[-1],train_size=0.8,stratify=data[-1],random_state=2024)
    best_score = 0
    for i in c_range:
        for g in g_range:
            svm_model = SVC(kernel='rbf',C=i,gamma=g)    
            cv = KFold(n_splits=5)
            accuracy = cross_val_score(svm_model,x_train, y_train, scoring='accuracy', cv=cv,n_jobs=-1)
            avg_score = accuracy.mean()
            if avg_score > best_score:
                best_score = avg_score
                best_parameters = {'gamma': g, "C": i}
    best_model = SVC(kernel='rbf',C=best_parameters["C"],gamma=best_parameters['gamma'])
    cls = best_model.fit(x_train,y_train)
    test_acc = cls.score(x_test,y_test)
    return best_parameters, best_score, test_acc


def res_RF(data,n_estimators):
    x_train,x_test,y_train,y_test = train_test_split(data[0],data[-1],train_size=0.8,stratify=data[-1],random_state=2024)
    best_score = 0
    for i in n_estimators:
        rf_model = RandomForestClassifier(n_estimators=i, max_features=2,criterion='gini')
        cv = KFold(n_splits=5)
        accuracy = cross_val_score(rf_model,x_train, y_train, scoring='accuracy', cv=cv,n_jobs=-1)
        avg_score = accuracy.mean()
        if avg_score > best_score:
            best_score = avg_score
            best_parameters = {'n_estimators': i}
    best_model = RandomForestClassifier(n_estimators=best_parameters["n_estimators"], max_features=2,criterion='gini')
    cls = best_model.fit(x_train,y_train)
    test_acc = cls.score(x_test,y_test)
    return best_parameters, best_score, test_acc

data = data_load('datasets.xlsx','Sheet1')
c_range = range(1,101,1)
g_range = range(1,11)
_, svm_best_score, svm_test_acc = res_SVC(data,c_range,g_range)

n_estimators = range(1,201,10)
_, rf_best_score, rf_test_acc = res_RF(data,n_estimators)


print('SVC train accuracy:'+str(svm_best_score))
print('SVC test accuracy:'+str(svm_test_acc))

print('RF best score:',rf_best_score)
print('RF test accuracy:',rf_test_acc)

