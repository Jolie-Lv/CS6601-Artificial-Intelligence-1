import unittest
import submission as dt
import numpy as np
import time
import pdb

def grid_search():
    num_trees_arr = [15]
    depth_limit_arr = [15]
    example_subsample_rate_arr = [0.7]
    attr_subsample_rate_arr = [0.7]

    for num_trees in num_trees_arr:
        for depth_limit in depth_limit_arr:
            for example_subsample_rate in example_subsample_rate_arr:
                for attr_subsample_rate in attr_subsample_rate_arr:
                    test_clf({'num_trees': num_trees,
                              'depth_limit': depth_limit,
                              'example_subsample_rate': example_subsample_rate,
                              'attr_subsample_rate': attr_subsample_rate})

def test_clf(params):
    dataset = dt.load_csv('challenge_train.csv', 0)
    # pdb.set_trace()
    train_features, train_classes = dataset
    folds = dt.generate_k_folds(dataset, 5)
    accuracy = []

    for idx, fold in enumerate(folds):
        training_set, test_set = fold
        clf = dt.ChallengeClassifier(**params)
        clf.fit(training_set[0], training_set[1])
        preds = clf.classify(test_set[0])
        accuracy.append(dt.accuracy(preds, test_set[1]))
        # print("Fold %d" %idx)
        # print("accuracy %f" %(dt.accuracy(preds, test_set[1])))
        # print("precision %f" %(dt.precision(preds, test_set[1])))
        # print("recall %f" %(dt.recall(preds, test_set[1])))
    print(params, np.mean(accuracy))



if __name__ == '__main__':
    grid_search()
