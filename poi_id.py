#!/usr/bin/python

import sys
import pickle
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
sys.path.append("../tools/")

### Selecting features
poi = ['poi']
fin_features = ['salary', 'deferral_payments', 'total_payments',\
                'loan_advances', 'bonus', 'restricted_stock_deferred',\
                'deferred_income', 'total_stock_value', 'expenses',\
                'exercised_stock_options', 'other', 'long_term_incentive',\
                'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages',\
                  'from_this_person_to_poi', 'shared_receipt_with_poi']
features_list = poi + fin_features + email_features
 
### Loading the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

    
### Removing outliers. "Total" and "THE TRAVEL AGENCY IN THE PARK" are not 
### persons.  "LOCKHART EUGENE E" has only empty fields
outliers = ['THE TRAVEL AGENCY IN THE PARK', 'TOTAL', 'LOCKHART EUGENE E']
for key in outliers:
    data_dict.pop(key)

    
### Creating new features
### "PCA1" is first PCA component of all financial features
### 'from_POI_percent' is percentage of emails from POI in all incoming emails
### 'to_POI_percent' is percentage of emails to POI in all ourcoming emails
financial_data = featureFormat(data_dict, poi + fin_features)
email_data = featureFormat(data_dict, poi + email_features, remove_NaN=False,\
                           remove_all_zeroes=False)
labels, fin_ftrs = targetFeatureSplit(financial_data)
labels, email_ftrs = targetFeatureSplit(email_data)
pca = PCA(n_components=1)
pca.fit(fin_ftrs)
financial_data = pca.transform(fin_ftrs)

### Devision function
def division(numerator, denominator):
    if denominator == 0 or np.isnan(numerator) or np.isnan(denominator):
        return 0
    return float(numerator) / denominator

### Adding new features to dataset
counter = 0
for key in data_dict:
    data_dict[key]['PCA1'] = financial_data[counter][0]   
    data_dict[key]['from_POI_percent'] = division(email_data[counter][1],\
                                                  email_data[counter][0])
    data_dict[key]['to_POI_percent'] = division(email_data[counter][3],\
                                                email_data[counter][2])   
    counter += 1


### Extracting all features from dataset and separating labels and features    
new_features = ['PCA1', 'from_POI_percent', 'to_POI_percent']
test_dataset= data_dict
test = featureFormat(test_dataset, features_list + new_features)
test_labels, test_features = targetFeatureSplit(test)


###Choosing 6 best features 
test_clf = SelectKBest(k=6)
test_clf.fit(test_features,test_labels)
### Printing features and scores
print test_clf.get_params()
for (f, v, s) in zip(features_list[1:] + new_features, test_clf.get_support(),\
                     test_clf.scores_):
    print v, f, s 


    
### 6 top features in my opinion. Result of personal thoughts and reviewing of
### SelectKBest choice  
features_list = poi + ['salary', 'exercised_stock_options', 'PCA1',\
                       'from_POI_percent', 'to_POI_percent']
my_dataset = data_dict


### Extracting features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Trying different classifiers. Comment after code is result of classifier.
### Naive Bayes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
clf_Gaus =  Pipeline([('scaler', MinMaxScaler()), ('GaussianNB', GaussianNB())])
clf_Gaus.fit(features, labels)
'''
GaussianNB()
Accuracy: 0.93140  Precision: 0.81546 Recall: 0.62750  F1: 0.70924  F2: 0.65783
Total predictions: 15000        True positives: 1255    False positives:  284  
False negatives:  745   True negatives: 12716
'''

### k-Nearest Neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
clf_KN = Pipeline([('scaler', MinMaxScaler()), ('KNeigh', KNeighborsClassifier())])
clf_KN.fit(features, labels)
'''
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
Accuracy: 0.88020  Precision: 0.62244 Recall: 0.25800  F1: 0.36479  F2: 0.29222
Total predictions: 15000        True positives:  516    False positives:  313
False negatives: 1484   True negatives: 12687
'''

### Logistic regression 
from sklearn.linear_model import LogisticRegression
clf_LR= Pipeline([('scaler', MinMaxScaler()), ('LogReg', LogisticRegression())])
clf_LR.fit(features, labels)
'''
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
Accuracy: 0.93220 Precision: 0.76902  Recall: 0.70250  F1: 0.73426  F2: 0.71487
Total predictions: 15000        True positives: 1405    False positives:  422   
False negatives:  595   True negatives: 12578
'''

### Support vector machines
from sklearn.svm import SVC
clf_SVM = Pipeline([('scaler', MinMaxScaler()), ('SVM', SVC())])
clf_SVM.fit(features, labels)
'''
Got a divide by zero when trying out: SVC(C=1.0, cache_size=200, 
class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, 
gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None,
shrinking=True, tol=0.001, verbose=False)
Precision or recall may be undefined due to a lack of true positive predicitons
'''

###Desision Trees
from sklearn.tree import DecisionTreeClassifier
clf_DT = Pipeline([('scaler', MinMaxScaler()), ('DTree', DecisionTreeClassifier())])
clf_DT.fit(features, labels)
'''
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
Accuracy: 0.94547  Precision: 0.80246  Recall: 0.78400  F1: 0.79312 F2: 0.78762
otal predictions: 15000        True positives: 1568    False positives:  386   
False negatives:  432   True negatives: 12614
'''

### Random forest
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
clf_RF = Pipeline([('scaler', MinMaxScaler()), ('RForest', RandomForestClassifier())])
clf_RF.fit(features, labels)
'''
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
Accuracy: 0.95427  Precision: 0.93167  Recall: 0.70900  F1: 0.80522 F2: 0.74459
Total predictions: 15000        True positives: 1418    False positives:  104   
False negatives:  582   True negatives: 12896
'''

### AdaBoost
clf_AB = Pipeline([('scaler', MinMaxScaler()), ('AdaBoost', AdaBoostClassifier())])
clf_AB.fit(features, labels)
'''
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=50, random_state=None)
Accuracy: 0.95020  Precision: 0.89930  Recall: 0.70550 F1: 0.79070  F2: 0.73728
Total predictions: 15000        True positives: 1411    False positives:  158   
False negatives:  589   True negatives: 12842
'''

### QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf_QDA = Pipeline([('scaler', MinMaxScaler()), ('QuadDisc', QuadraticDiscriminantAnalysis())])
clf_QDA.fit(features, labels)
'''
QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
               store_covariances=False, tol=0.0001)
Accuracy: 0.77553  Precision: 0.14931  Recall: 0.14550 F1: 0.14738  F2: 0.14625
Total predictions: 15000        True positives:  291    False positives: 1658   
False negatives: 1709   True negatives: 11342
'''


###Tuning 2 best classifiers to achieve better results 
###Decision Tree
from sklearn.grid_search import GridSearchCV
parameters = {'criterion': ('gini','entropy'), 'splitter': ('best', 'random'),\
              'max_features':('sqrt', 'log2', None),\
              'min_samples_leaf':[1, 5, 10, 15]}
clf_DT = DecisionTreeClassifier()
clf_DT_best = GridSearchCV(clf_DT, parameters)
clf_DT_best.fit(features, labels)
print clf_DT_best.best_params_

###Decision Tree using best parameters
clf_DT_best =  Pipeline([('scaler', MinMaxScaler()), \
               ('BestDT', DecisionTreeClassifier(criterion = 'entropy', \
                                                  splitter= 'best',\
                                     max_features= None, min_samples_leaf= 1))]) 
clf_DT_best.fit(features, labels)


'''
Best results from parameters:
{'criterion': 'gini',
 'max_features': None,
 'min_samples_leaf': 1,
 'splitter': 'random'}    

GridSearchCV(cv=None, error_score='raise',
   estimator=DecisionTreeClassifier(class_weight=None, criterion='gini',
   max_depth=None, max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
   min_samples_split=2, min_weight_fraction_leaf=0.0,
   presort=False, random_state=None, splitter='best'),
       fit_params={}, iid=True, n_jobs=1,
param_grid={'max_features': ('sqrt', 'log2', None), 
'splitter': ('best', 'random'), 'criterion': ('gini', 'entropy'),
'min_samples_leaf': [1, 5, 10, 15]},pre_dispatch='2*n_jobs', 
refit=True, scoring=None, verbose=0)
Accuracy: 0.94667  Precision: 0.87129  Recall: 0.70400 F1: 0.77876  F2: 0.73211
Total predictions: 15000        True positives: 1408    False positives:  208   
False negatives:  592   True negatives: 12792
'''

### Random Forest
parameters = {'n_estimators': [2, 5, 10, 15, 20],\
              'criterion': ('gini','entropy'),\
              'max_features':('sqrt', 'log2', None),\
              'min_samples_leaf':[1, 5, 10, 15]}
clf_RF = RandomForestClassifier()
clf_RF_best = GridSearchCV(clf_RF, parameters)
clf_RF_best.fit(features, labels)
print clf_RF_best.best_params_
'''
Best results from parameters:
{'criterion': 'gini',
 'max_features': 'log2',
 'min_samples_leaf': 1,
 'n_estimators': 15}
 
GridSearchCV(cv=None, error_score='raise',
  estimator=RandomForestClassifier(bootstrap=True, class_weight=None,
  criterion='gini', max_depth=None, max_features='auto', max_leaf_nodes=None,
  min_samples_leaf=1, min_samples_split=2,
  min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
  oob_score=False, random_state=None, verbose=0, warm_start=False),
       fit_params={}, iid=True, n_jobs=1,
       param_grid={'n_estimators': [2, 5, 10, 15, 20],
       'max_features': ('sqrt', 'log2', None),
       'criterion': ('gini', 'entropy'), 'min_samples_leaf': [1, 5, 10, 15]},
       pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=0)
Accuracy: 0.95500  Precision: 0.93047  Recall: 0.71600  F1: 0.80927 F2: 0.75060
Total predictions: 15000        True positives: 1432    False positives:  107   
False negatives:  568   True negatives: 12893
'''

###Random Forest using best parameters
clf_RF_best =  Pipeline([('scaler', MinMaxScaler()), \
               ('BestRF', RandomForestClassifier(criterion = 'entropy', \
                                    max_features= None, min_samples_leaf= 1,
                                    n_estimators = 15))]) 
clf_RF_best.fit(features, labels)


### Setting best result
clf = clf_RF_best

### Dumping results
dump_classifier_and_data(clf, my_dataset, features_list)
