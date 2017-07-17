import sys
from prep_terrain_data import makeTerrainData
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

import numpy as np

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = AdaBoostClassifier()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)