Falsely-classified instances (false positives + false negatives) from classifier_test_result, on classifiers with 5-fold training and testing data split (no Grid Search):
1. MLPClassifier: 2 layers, 32 neurons per layer. Histogramizer 6 bins, (0, Max's post-threshold)
3. Logistic Regression: l2 regularization, C = 10**-3
2. Random Forest

classifier_test_result_mlp: 5-fold training/testing split, no Grid Search. MLPClassifier: 2 layers, 32 neurons per layer. Histogramizer 6 bins, (0, Max's post-threshold)
classifier_test_result_mlp_gs: 5-fold training/testing split, Grid Search. MLPClassifier, 2 layers, 8-40 neurons per layer (the paper's method including the Grid Search part)
