from sklearn.model_selection import KFold,cross_val_score,cross_val_predict
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

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

data = data_load('code/biyelunwen/形貌数据.xlsx','Sheet1')
c_range = range(1,101,1)
g_range = range(1,11)
_, svm_best_score, svm_test_acc = res_SVC(data,c_range,g_range)

n_estimators = range(1,201,10)
_, rf_best_score, rf_test_acc = res_RF(data,n_estimators)


print('SVC train accuracy:'+str(svm_best_score))
print('SVC test accuracy:'+str(svm_test_acc))

print('RF best score:',rf_best_score)
print('RF test accuracy:',rf_test_acc)








# def main():
#     data = data_load('code/biyelunwen/形貌数据.xlsx','Sheet1')
#     x_train,x_test,y_train,y_test = train_test_split(data[0],data[-1],train_size=0.8,stratify=data[-1],random_state=2024)
#     def select_c_function(i):
#         svm_model = SVC(kernel='rbf',C=i,gamma=g)
#         # rf_model = RandomForestClassifier(n_estimators=i, max_features=2,criterion='gini')
#         cv=KFold(n_splits=5)
#         accuracy = cross_val_score(svm_model,x_train, y_train, scoring='accuracy', cv=cv,n_jobs=-1)
#         return accuracy.mean()
#     best_score = 0
#     c_range = range(1,101,1)
#     g_range = range(1,11)
#     for i in c_range:
#         for g in g_range:
#             avg_score = select_c_function(i)
#             if avg_score > best_score:
#                 best_score = avg_score
#                 best_parameters = {'gamma': g, "C": i}
#                 # best_parameters = {'n_estimators': i}
#     print(best_parameters,'最佳准确率是'+str(best_score))
#     a = best_parameters["C"]    
#     b = best_parameters['gamma']
#     # print('n_estimators:'+str(a))#,'gama:'+str(b))
#     last_mode = SVC(kernel='rbf',C=a,gamma=b)
#     # last_mode = RandomForestClassifier(n_estimators=a, max_features=2,criterion='gini')
#     model = last_mode.fit(x_train,y_train)
#     # train_acc = model.score(x_train,y_train)
#     # y_pre_train = model.predict(x_train)
#     # train_tpr = recall_score(y_train, y_pre_train)
#     test_acc = model.score(x_test,y_test)
#     # y_pre_test = model.predict(x_test)
#     # test_tpr = recall_score(y_test,y_pre_test)
#     # print('训练集准确率：'+str(train_acc),'训练集tpr:'+str(train_tpr))
#     print('测试集准确率：'+str(test_acc))#,'测试集tpr:'+str(test_tpr))

#     # svm_model = SVC(kernel='rbf',C=a,gamma=b)
#     # # rf_model = RandomForestClassifier(n_estimators=a, max_features=2,criterion='gini')
#     # cv=KFold(n_splits=5)
#     # y_pre = cross_val_predict(svm_model,data[0],data[-1], cv=cv)
#     # conf_matrix = confusion_matrix(data[-1],y_pre)
#     # heat_reg=sns.heatmap(conf_matrix,annot=True,fmt='.20g',square=True,linewidths=0.05,annot_kws={"fontsize":18})
#     # plt.rcParams['font.sans-serif']=['SimHei']
#     # plt.rcParams['axes.unicode_minus'] = False
#     # plt.gca().set_xticklabels(['银线','晶核','颗粒','混合'],fontsize=18)#输出不同时需调整标签
#     # plt.gca().set_yticklabels(['银线','晶核','颗粒','混合'],fontsize=18)#输出不同时需调整标签
#     # plt.show()

# main()
