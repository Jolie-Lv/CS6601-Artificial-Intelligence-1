# CS6601 Fall 2019, Assignment 4 Kaggle competition

from submission import ChallengeClassifier, generate_k_folds, accuracy
import numpy as np


def loaddata():
    my_data = np.genfromtxt('kaggle_train_2019_fall.csv', delimiter=',')
    classes = my_data[:,0].astype(int)
    features = my_data[:,1:]
    return features, classes


def accuracy_test(num_folds):
    features, classes = loaddata()
    folds = generate_k_folds((features, classes), num_folds)
    for i in range(num_folds):
        print('Testing model on fold %d' % i)
        myClassifier = ChallengeClassifier()
        training_data, test_data = folds[i]
        training_features, training_classes = training_data
        test_features, test_classes = test_data
        myClassifier.fit(training_features, training_classes)
        result = myClassifier.classify(test_features)
        print('Accuracy test result - fold %d: %f' % (i, accuracy(result, test_classes)))


def generate_kaggle_submission():
    features, classes = loaddata()
    test_data = np.genfromtxt('kaggle_test_2019_fall_unlabelled.csv', delimiter=',')
    myClassifier = ChallengeClassifier()
    myClassifier.fit(features, classes)
    result = myClassifier.classify(test_data)

    result_with_id = np.array([range(0,len(result)), result]).transpose()
    np.savetxt("kaggle_result.csv", result_with_id, fmt='%d', delimiter=",", header = "Id,Class")


if __name__ == "__main__":
    ###cross-validation - remove following line if you just wanna generate results to upload on Kaggle
    accuracy_test(num_folds=2)
    ###function to generate results to upload on Kaggle
    generate_kaggle_submission()
