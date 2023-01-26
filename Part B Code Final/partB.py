import pandas as pd
import os
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import en_core_web_sm


nlp = en_core_web_sm.load()

snowball = SnowballStemmer(language='english')

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([snowball.stem(w) for w in analyzer(doc)])


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: ([snowball.stem(w) for w in analyzer(doc)])

prohibitedWords = stopwords.words('english') #getting the stopwords from nltk
big_regex = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, reversed(prohibitedWords)))) #regex to remove stopwords


file_list=[]
df_train = pd.DataFrame(columns=['doc_id', 'class', 'text']) # dataframe for the training data
df_test = pd.DataFrame(columns=['doc_id', 'class', 'text']) # dataframe for the testing data

counter = 1 # for the doc id
paths = ['/home/athinafus/Documents/LinguisticTech/lingtech/lingtech/20news-bydate-test/',
         '/home/athinafus/Documents/LinguisticTech/lingtech/lingtech/20news-bydate-train/']


for path in paths:
    spl = path.split('/')[7]
    for folder in os.listdir(path):
        for file in os.listdir("{}/{}".format(path, folder)):
            with open(os.path.join("{}/{}".format(path, folder), file), 'r') as f:
                people_names=[]
                try:
                    text = f.read()
                except:
                    continue

                res = re.split("Lines: [1-9][0-9]", text) # removing lines before "Lines: <number of lines>
                try:
                    text = res[1]
                except:
                    try:
                        res = re.split("Lines: [1-9]", text)  # removing lines before "Lines: <number of lines>
                        text = res[1]
                    except:
                        continue
                doc = nlp(text)
                if len(doc.ents) != 0:  # if there are named entities in the text
                    # -----[f(x) for x in sequence if condition]----------------------------------------------------------------
                    [people_names.append(X.text) for X in doc.ents
                     if X.label_ in ['PERSON', 'QUANTITY', 'DATE', 'TIME', 'PERCENT', 'CARDINAL']]
                else:
                    print('No named entities in text')
                name_regex = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, reversed(people_names))))
                text = name_regex.sub("", text)  # removing the names from the text
                text = re.sub(r'[^\w]', ' ', text) # removing symbols and empty lines
                text = text.replace("_", ' ')
                text = big_regex.sub("", text) # removing the stopwords from the text
                text = ''.join([i for i in text if not i.isdigit()]) # removing numbers from the text
                text = re.sub(' +', ' ',text) # removing too many spaces from text

                #print("FILE: {}-------------FOLDER:{}------SET:{}--------------------------".format(file, folder,path.split('/')[7].split('-')[2]))
                #print("FILE: {}-------------after clean--------------------------------".format(file))
                if path.split('/')[7] == '20news-bydate-train':
                    df_train = df_train.append({'doc_id': 'd{}'.format(counter), 'class': folder, 'text': str(text)},ignore_index=True)
                else:
                    df_test = df_test.append({'doc_id': 'd{}'.format(counter), 'class': folder, 'text': str(text)},
                                   ignore_index=True)
                counter += 1

print("TRAINING DATAFRAME")
print(df_train.to_string)
print("TESTING DATAFRAME")
print(df_test.to_string)

scv = StemmedCountVectorizer(max_features=12000) # setting max number of features we want to use for classification
stfidfvectorizer = StemmedTfidfVectorizer(max_features=12000) # setting max number of features we want to use for classification
                                                #build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.


#===========================START OF TRAINING-DATAFRAME==================================================================
count_wm_TRAIN = scv.fit_transform(df_train['text'])
stfidf_wm_TRAIN = stfidfvectorizer.fit_transform(df_train['text'])#retrieve the terms found in the corpora

count_tokens = scv.get_feature_names_out()
stfidf_tokens = stfidfvectorizer.get_feature_names_out()

df_countvect_TRAIN = pd.DataFrame(data = count_wm_TRAIN.toarray(),columns = count_tokens)
sdf_tfidfvect_TRAIN = pd.DataFrame(data = stfidf_wm_TRAIN.toarray(),columns = stfidf_tokens) #contains the documents as vectors and
                                                                              # each element is the tf-idf weight of each
                                                                              #stem

print('<<<<<<<<TRAIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print("Count Vectorizer\n")
print(df_countvect_TRAIN)
'''print("\nTD-IDF Vectorizer\n")
print(df_tfidfvect_TRAIN)'''
print("\nSTf-IDF Vectorizer\n")
print(sdf_tfidfvect_TRAIN)
# now calculate the sum, call sort on the series
s = sdf_tfidfvect_TRAIN.sum().sort_values(ascending=False, inplace=False)
# now use fancy indexing to reorder the df
sdf_tfidfvect_TRAIN = sdf_tfidfvect_TRAIN.reindex(s.index, axis=1).dropna(how='all', axis=1) # can set max_features and keep the
                                                                                 # n features with max tf-idf weights

#===========================END OF TRAINING-DATAFRAME===================================================================



#===========================START OF TESTING-DATAFRAME==================================================================
count_wm_TEST = scv.fit_transform(df_test['text'])
stfidf_wm_TEST = stfidfvectorizer.fit_transform(df_test['text'])#retrieve the terms found in the corpora

count_tokens = scv.get_feature_names_out()
stfidf_tokens = stfidfvectorizer.get_feature_names_out()

df_countvect_TEST = pd.DataFrame(data = count_wm_TEST.toarray(),columns = count_tokens)
sdf_tfidfvect_TEST = pd.DataFrame(data = stfidf_wm_TEST.toarray(),columns = stfidf_tokens) #contains the documents as vectors and
                                                                              # each element is the tf-idf weight of each
                                                                              #stem

print("Count Vectorizer\n")
print(df_countvect_TEST)
'''print("\nTD-IDF Vectorizer\n")
print(df_tfidfvect_TEST)'''
print("\nSTf-IDF Vectorizer\n")
print(sdf_tfidfvect_TEST)
print('SUM:')
print(sdf_tfidfvect_TEST.sum(axis = 0, skipna = True)) #sum of the tf-idf weights for every stem
print(sdf_tfidfvect_TEST.head())
# now calculate the sum, call sort on the series
s = sdf_tfidfvect_TEST.sum().sort_values(ascending=False, inplace=False)
# now use fancy indexing to reorder the df
sdf_tfidfvect_TEST = sdf_tfidfvect_TEST.reindex(s.index, axis=1).dropna(how='all', axis=1) # can set max_features and keep the
                                                                                 # n features with max tf-idf weights
#===========================END OF TESTING-DATAFRAME====================================================================


# ==================INTERSECTIONING THE STEMS IN EACH DATAFRAME=========================================================
stems_intersection = list(set(sdf_tfidfvect_TRAIN.columns) & set(sdf_tfidfvect_TEST.columns)) # find intersection of columns

sdf_tfidfvect_TRAIN = sdf_tfidfvect_TRAIN.reindex(stems_intersection, axis=1).dropna(how='all', axis=1)
sdf_tfidfvect_TEST = sdf_tfidfvect_TEST.reindex(stems_intersection, axis=1).dropna(how='all', axis=1)

arr_TEST = sdf_tfidfvect_TEST.values
arr_TRAIN = sdf_tfidfvect_TRAIN.values


sdf_tfidfvect_TEST['Vector']  = pd.DataFrame([[i] for i in arr_TEST])   #                               +++
sdf_tfidfvect_TRAIN['Vector']  = pd.DataFrame([[i] for i in arr_TRAIN])   #                             $$$


sdf_tfidfvect_TEST.insert(0, "Doc_id", df_test['doc_id'], True) # adding the doc id AFTER               +++
sdf_tfidfvect_TEST.insert(1, "Class", df_test['class'], True) # adding the class AFTER                 +++

sdf_tfidfvect_TRAIN.insert(0, "Doc_id", df_train['doc_id'], True) # adding the doc id AFTER             $$$
sdf_tfidfvect_TRAIN.insert(1, "Class", df_train['class'], True) # adding the class AFTER               $$$


print('<<<<<<TRAIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

print("\nSTf-IDF Vectorizer -top 100\n")
print(sdf_tfidfvect_TRAIN.head().to_string())
print(sdf_tfidfvect_TRAIN.info(verbose=True))
print(sdf_tfidfvect_TRAIN.info(verbose=True, null_counts=True))

print('<<<<<<TEST<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

print("\nSTf-IDF Vectorizer -top 100\n")
print(sdf_tfidfvect_TEST.head().to_string())
print(sdf_tfidfvect_TEST.info(verbose=True, null_counts=True))
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

# Exporting the tf-idf matrices to pickle files, for faster access to the data
sdf_tfidfvect_TRAIN.to_pickle('pickles_vectors/sdf_tfidfvect_train8000.pickle')
sdf_tfidfvect_TEST.to_pickle('pickles_vectors/sdf_tfidfvect_test8000.pickle')

# print out the statistics of the classes in the training set and the testing set
print("Training stats:\n{}".format(sdf_tfidfvect_TRAIN["Class"].value_counts(normalize=True)))
print("Testing stats:\n{}".format(sdf_tfidfvect_TEST["Class"].value_counts(normalize=True)))

