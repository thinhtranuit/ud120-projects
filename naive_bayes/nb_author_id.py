#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

from time import time
from tools.email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
# your code goes here ###
clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print("Training time: {:.2f}".format(time()-t0))
GaussianNB(priors=None, var_smoothing=1e-09)
t1 = time();
predict = clf.predict(features_test)
print("Predicting time: {:.2f}".format(time()-t1))
print(predict)
score = clf.score(features_test, labels_test)
print(score)

#########################################################


