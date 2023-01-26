import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers


sdf_tfidfvect_TRAIN = pd.read_pickle('pickles_vectors/sdf_tfidfvect_train8000.pickle')
sdf_tfidfvect_TEST = pd.read_pickle('pickles_vectors/sdf_tfidfvect_test8000.pickle')

print("\nSTf-IDF Vectorizer -top 100 ---TRAIN\n")
print(sdf_tfidfvect_TRAIN.head().to_string())
print(sdf_tfidfvect_TRAIN.info(verbose=True, show_counts=True))
print('<<<<<<TEST<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

print("\nSTf-IDF Vectorizer -top 100 ---TEST\n")
print(sdf_tfidfvect_TEST.head().to_string())
print(sdf_tfidfvect_TEST.info(verbose=True, show_counts=True))
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')


# =START: Plan no. 1====================================================================================================
#Train_X = sdf_tfidfvect_TRAIN["Vector"].values
Train_Y = sdf_tfidfvect_TRAIN["Class"].values
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
Train_X = sdf_tfidfvect_TRAIN.iloc[:,2:9189].values

print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print(Train_X)

Test_X = sdf_tfidfvect_TEST.iloc[:,2:9189].values
Test_Y = sdf_tfidfvect_TEST["Class"].values

Encoder = LabelEncoder() # to encode the target variable, because it's string and the model will not understand it
Train_Y = Encoder.fit_transform(Train_Y)

Test_Y = Encoder.transform(Test_Y)

print("LENGTH of vector: {}".format(len(Train_X[0])))



'''model.add(Dense(12, input_dim=3425, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(20, activation='softmax'))'''

model = Sequential()
model.add(Dense(12, input_dim=9187, activation='relu'))
model.add(layers.Dense(80)) # no activation here
model.add(layers.LeakyReLU(alpha=0.3)) # activation layer here instead
model.add(layers.Dropout(0.2))
model.add(layers.Dense(60)) # no activation here
model.add(layers.LeakyReLU(alpha=0.3))  # activation layer here instead
model.add(layers.Dropout(0.2))
model.add(layers.Dense(20, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=[['accuracy']])

model.fit(Train_X, Train_Y, epochs=100, batch_size=30)
_, accuracy = model.evaluate(Train_X, Train_Y)
print('Accuracy: %.2f' % (accuracy*100))
# evaluate the keras model
'''loss, accuracy, f1_score, precision, recall = model.evaluate(Test_X, Test_Y, verbose=1)
print('Accuracy: %f' % accuracy)
print('Precision: %f' % precision)
print('Recall: %f' % recall)
print('F1 score: %f' % f1_score)'''

#predictions = model.predict(Test_X)
print(Test_X)
score = model.evaluate(Test_X, Test_Y, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

score = model.evaluate(Train_X, Train_Y, verbose=0)
print(f'Train loss: {score[0]} / Train accuracy: {score[1]}')
