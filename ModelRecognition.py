__author__ = 'tbulgerin'

import numpy
from sklearn import svm
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import LeaveOneOut
from sklearn import cross_validation as cval
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

import FeatureExtraction

def main():
    # perform feature extraction to get a file containing
    # the features of all models
    FeatureExtraction.featureExtraction()

    # load the features
    X, Y = load_svmlight_file("features/features.txt")

    # create the classifier and set the gamma value
    clf = svm.SVC(gamma=0.0001, C=100)

    # set the cross validation to be leave-one-out
    cv = LeaveOneOut(len(Y))

    # exhaustive grid search
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.005, .0005],
    #                      'C': [1, 10, 100, 1000]}]
    # clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=cv, scoring='accuracy')
    # clf.fit(X, Y)
    # print(clf.best_estimator_)
    # for params, mean_score, scores in clf.grid_scores_:
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean_score, scores.std() / 2, params))

    # calculate the accuracy
    scores = cval.cross_val_score(clf, X, Y, cv=cv, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # calculate confusion matrix/ROC curve
    # labels = [1, 2, 3, 4, 5, 6, 7]
    # confusionMatrix = numpy.zeros((len(labels), len(labels)), dtype=numpy.int)
    # for train, test in cv:
    #     X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
    #     y_pred = clf.fit(X_train, y_train).predict(X_test)
    #     confusionMatrix = confusionMatrix + confusion_matrix(y_test, y_pred, labels)
    #
    # print confusionMatrix


if __name__ == '__main__':
    main()
