from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    accuracy = accuracy_score(pred, labels_test)
    return accuracy