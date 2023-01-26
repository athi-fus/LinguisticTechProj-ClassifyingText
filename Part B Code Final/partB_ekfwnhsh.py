import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from sklearn.metrics import accuracy_score



sdf_tfidfvect_TRAIN = pd.read_pickle('pickles_vectors/sdf_tfidfvect_train8000.pickle')
sdf_tfidfvect_TEST = pd.read_pickle('pickles_vectors/sdf_tfidfvect_test8000.pickle')

print("\nSTf-IDF Vectorizer -top 100 ---TRAIN\n")
print(sdf_tfidfvect_TRAIN.head().to_string())
print(sdf_tfidfvect_TRAIN.info(verbose=True, show_counts=True))
print('<<<<<<TEST<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

print("\nSTf-IDF Vectorizer -top 100 ---TEST\n")
#print(sdf_tfidfvect_TEST.head().to_string())
print(sdf_tfidfvect_TEST.info(verbose=True, show_counts=True))
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

# Using cosine similarity to classify documents
# Plan: For each class from the training set, calculate a mean-vector and compare the on-test vector with these
# =START: Plan==========================================================================================================
print('\n\nMEAN')
print(sdf_tfidfvect_TRAIN.groupby(['Class']).mean())
sdf_tfidfvect_TRAIN_mean = sdf_tfidfvect_TRAIN.groupby(['Class']).mean()

# ==========================NEW DATAFRAMES==============================================================================
# For Cosine Similarity
similar = pd.DataFrame(columns=['Document', 'Original_class', 'Assigned_class', 'Cosine_similarity'])
# For Manhattan Distance
df_Manhattan = pd.DataFrame(columns=['Document', 'Original_class', 'Assigned_class', 'Manhattan_Distance'])
# For Euclidean Distance
df_Euclidean = pd.DataFrame(columns=['Document', 'Original_class', 'Assigned_class', 'Euclidean_Distance'])



for index_test, row_test in sdf_tfidfvect_TEST.iterrows():
    temp_max = 0
    temp_min_m = 100000
    temp_min_e = 100000
    for index, row in sdf_tfidfvect_TRAIN_mean.iterrows():
        vec_train = row.values
        cs = cosine_similarity(row_test['Vector'].reshape(1, -1), vec_train.reshape(1, -1))
        dst_m = distance.cityblock(row_test['Vector'], vec_train)
        dst_e = distance.euclidean(row_test['Vector'], vec_train)

        if cs[0][0] > temp_max: # cs[0][0] is the cosine similarity score
            temp_max = cs[0][0]
            max_class = index

        if dst_m < temp_min_m:  # dst_m the manhattan distance
            temp_min_m = dst_m
            max_class_m = index

        if dst_e < temp_min_e:  # dst_e is the euclidean distance
            temp_min_e = dst_e
            max_class_e = index

    similar = similar.append({"Document": row_test['Doc_id'], 'Original_class': row_test['Class'],
                              'Assigned_class': max_class, 'Cosine_similarity': temp_max},ignore_index=True )
    df_Manhattan = df_Manhattan.append({"Document": row_test['Doc_id'], 'Original_class': row_test['Class'],
                                        'Assigned_class': max_class_m, 'Manhattan_Distance': temp_min_m},
                                       ignore_index=True)
    df_Euclidean = df_Euclidean.append({"Document": row_test['Doc_id'], 'Original_class': row_test['Class'],
                              'Assigned_class': max_class_e, 'Euclidean_Distance': temp_min_e},ignore_index=True )

# =END: Plan no. 1======================================================================================================


print(similar.head(100).to_string())
print('\n')
print(df_Manhattan.head(100).to_string())
print('\n')
print(df_Euclidean.head(100).to_string())
print("\nCosine Similarity: Accuracy Score -> ", accuracy_score(similar['Assigned_class'], similar['Original_class'])*100)
print("\nManhattan Distance: Accuracy Score -> ", accuracy_score(df_Manhattan['Assigned_class'],
                                                                 df_Manhattan['Original_class'])*100)
print("\nEuclidean Distance: Accuracy Score -> ", accuracy_score(df_Euclidean['Assigned_class'], df_Euclidean['Original_class'])*100)
