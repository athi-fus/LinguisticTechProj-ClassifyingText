import pandas as pd
from sklearn import naive_bayes, svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV

sdf_tfidfvect_TRAIN = pd.read_pickle('pickles_vectors/sdf_tfidfvect_train8000.pickle')
sdf_tfidfvect_TEST = pd.read_pickle('pickles_vectors/sdf_tfidfvect_test8000.pickle')

print('<<<<<<TRAIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

print("\nSTf-IDF Vectorizer -top 100\n")
print(sdf_tfidfvect_TRAIN.head().to_string())
print(sdf_tfidfvect_TRAIN.info(verbose=True))
print(sdf_tfidfvect_TRAIN.info(verbose=True, show_counts=True))
print('<<<<<<TEST<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

print("\nSTf-IDF Vectorizer -top 100\n")
print(sdf_tfidfvect_TEST.head().to_string())
print(sdf_tfidfvect_TEST.info(verbose=True, show_counts=True))
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

# ===========================MACHINE LEARNING===========================================================================

# print out the statistics of the classes in the training set and the testing set
print("Training stats:\n{}".format(sdf_tfidfvect_TRAIN["Class"].value_counts(normalize=True)))
print("Testing stats:\n{}".format(sdf_tfidfvect_TEST["Class"].value_counts(normalize=True)))


stems_intersection =  list(set(sdf_tfidfvect_TRAIN.columns) & set(sdf_tfidfvect_TEST.columns))
Train_X = sdf_tfidfvect_TRAIN["Vector"].tolist()
Train_Y = sdf_tfidfvect_TRAIN["Class"].tolist()

Test_X = sdf_tfidfvect_TEST["Vector"].tolist()
Test_Y = sdf_tfidfvect_TEST["Class"].tolist()

Encoder = LabelEncoder() # to encode the target variable, because it's string and the model will not understand it
Train_Y = Encoder.fit_transform(Train_Y)

Test_Y = Encoder.transform(Test_Y)

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X,Train_Y)# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X)# Use accuracy_score function to get the accuracy

#print(classification_report(Test_Y, predictions_NB))
print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y)*100)
print("Precision score: {}".format(precision_score(Test_Y,predictions_NB, average='macro')))
print("Recall score: {}".format(recall_score(Test_Y,predictions_NB, average='macro')))
print("F1 Score: {}\n".format(f1_score(Test_Y,predictions_NB, average='macro')))


# ====================SVM MODEL=========================================================================================
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, max_iter=3000)
print('Created the Linear Kernel SVM')
SVM.fit(Train_X,Train_Y)# predict the labels on validation dataset
print('Fitted the Linear Kernel SVM')
predictions_SVM = SVM.predict(Test_X)# Use accuracy_score function to get the accuracy

#print(classification_report(Test_Y, predictions_SVM))
print("\nLinear Kernel SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
print("Precision score: {}".format(precision_score(Test_Y,predictions_SVM, average='macro')))
print("Recall score: {}".format(recall_score(Test_Y,predictions_SVM, average='macro')))
print("F1 Score: {}\n".format(f1_score(Test_Y,predictions_SVM, average='macro')))


# cross validation =====================================================================================================
print("\n\n...Starting cross validation...")
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)
# fitting the model for grid search
grid.fit(Train_X, Train_Y)
print(grid.best_params_)
print(grid.best_estimator_)

SVM = svm.SVC(C=10.0, gamma=1.0)
print('Created the RBF Kernel SVM')
SVM.fit(Train_X,Train_Y)# predict the labels on validation dataset
print('Fitted the RBF Kernel SVM')
predictions_SVM = SVM.predict(Test_X)# Use accuracy_score function to get the accuracy

#print(classification_report(Test_Y, predictions_SVM))
print("\nRBF Kernel SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
print("Precision score: {}".format(precision_score(Test_Y,predictions_SVM, average='macro')))
print("Recall score: {}".format(recall_score(Test_Y,predictions_SVM, average='macro')))
print("F1 Score: {}".format(f1_score(Test_Y,predictions_SVM, average='macro')))





