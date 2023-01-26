import pickle
from pymongo import MongoClient
from operator import itemgetter
import time
import nltk
from nltk.stem import WordNetLemmatizer
import re

def lematize(word_list):
    for i in range(0, len(word_list)):
        lemm = wordnet_lemmatizer.lemmatize(word_list[i])
        word_list[i] = lemm
    return  word_list

def Extract(lst):
    return list(map(itemgetter(0), lst))

client = MongoClient("mongodb+srv://athinafus:hannahm0ng0@lingtech.bv1wl.mongodb.net/Lingtech?retryWrites=true&w=majority")
db = client['Article_Data']
collection = db['Articles']

a_file = open("inversed_file20_jan.pkl", "rb")
inverted_file = pickle.load(a_file)
a_file.close()

one_word=[ 'tiktok',
              'step',
              'movie',
              'long',
              'trump',
              'fire',
              'colombian',
              'impede',
              'point',
              'score',
              'telescope',
              'devastatingly',
              'autobiography',
              'autoimmune',
              'catering',
              'contradiction',
              'lucrative',
              'sentient',
              'ignore',
              'odyssey',
              'reactionary'
            ]
wordnet_lemmatizer = WordNetLemmatizer()

# ~~~~~~~~~~~~~~~~~~reading the 2-word queries from a file~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
two_words=[]
file0 = open('/home/athinafus/Documents/LinguisticTech/lingtech/two_word_queries.txt', 'r')
Lines = file0.readlines()
count = 0
# Strips the newline character
for line in Lines:
    count += 1
    #print("Line{}: {}".format(count, line.strip()))
    line = line.replace('\n', '').split(' ')
    #print(line)
    line = [x.lower() for x in line]
    line = lematize(line)
    two_words.append(line)

file0.close()
#print(three_words)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~reading the 3-word queries from a file~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
three_words=[]
file1 = open('/home/athinafus/Documents/LinguisticTech/lingtech/three_word_queries.txt', 'r')
Lines = file1.readlines()
count = 0
# Strips the newline character
for line in Lines:
    count += 1
    #print("Line{}: {}".format(count, line.strip()))
    line = line.replace('\n', '').split(' ')
    #print(line)
    line = [x.lower() for x in line]
    line = lematize(line)
    three_words.append(line)

file1.close()
#print(three_words)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~reading the 4-word queries from a file~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
four_words = []
file2 = open('/home/athinafus/Documents/LinguisticTech/lingtech/four_word_queries.txt', 'r') #reading the 4-word queries from a file
Lines = file2.readlines()
count = 0
# Strips the newline character
for line in Lines:
    count += 1
    #print("Line{}: {}".format(count, line.strip()))
    line = line.replace('\n', '').split(' ')
    line = [x.lower() for x in line]
    line = lematize(line)
    four_words.append(line)

file1.close()
#print(four_words)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
''' RANDOM WORD QUERIES
for k in range(0,30):
    #w1, w2, w3 = random.choice(list(inverted_file.items()))
    keys3 = random.sample(list(inverted_file), 3)
    three_words[k] = keys3

for j in range(0,30):
    #w1, w2, w3, w4 = random.choice(list(inverted_file.items()))
    keys4 = random.sample(list(inverted_file), 4)
    four_words[j] = keys4
'''


start = time.time() # start time for one word queries
for lem in one_word:
    #print("LEMMA: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(lem))
    start = time.time()
    doc_list = inverted_file[lem]  # list of documents that contain the lemma
    sorted_tfidf = sorted(doc_list, key=lambda x: x[1], reverse=True)  # list sorted based on tf-idf score

    #print("Lemma: {}".format(lem))
    #for item in sorted_tfidf:
        #print("\t{}".format(item))
        #x = collection.find_one({'_id': item[0]}, {"Path": 0, 'Article': 0})
        #print(x)

end = time.time()


print('Elapsed time for 1-word queries: {} ms'.format((end - start)*1000/20))

'''for i in range(0, len(one_word)): # building the 2-word queries
    two_words[i] = [one_word[i], one_word[i-1]]
'''


all_sets = [two_words, three_words, four_words]
for query_set in all_sets:
    #print("query set: {}".format(query_set))

    start = time.time()
    on_test_lemmas = {} # empty dictonary
    for w_set in query_set:
        doc_list = []
        for lem in w_set:

            try:
                #print(inverted_file[lem])
                doc_list.append(sorted(inverted_file[lem], key=lambda x: x[1], reverse=True))
                #sorted_tfidf = sorted(inverted_file[lem], key=lambda x: x[1], reverse=True)
            except Exception as e:
                print(e)  # printing the element that is causing the error
                print('LIST ERROR')

        if len(doc_list) == 2:
            tfidf={}
            intersect = list(set(Extract(doc_list[0])) & set(Extract(doc_list[1])) )
            for docs in intersect:
                for i in range(0, len(doc_list)):
                    for item in doc_list[i]:
                        #print(docs)
                        #print(item)
                        exists = bool(docs in item)
                        if exists:
                            try:
                                idx = doc_list[i].index(item)
                                #print(idx)
                                tfidf[docs]+=float(doc_list[i][idx][1])
                            except:
                                tfidf[docs] = doc_list[i][idx][1]
            end = time.time()

        elif len(doc_list) == 3:
            tfidf = {}
            intersect = list(set(Extract(doc_list[0])) & set(Extract(doc_list[1])) & set(Extract(doc_list[2])))
            for docs in intersect:
                for i in range(0, len(doc_list)):
                    for item in doc_list[i]:
                        #print(item)
                        exists = bool(docs in item)
                        if exists:
                            try:
                                idx = doc_list[i].index(item)
                                # print(idx)
                                tfidf[docs] += float(doc_list[i][idx][1])
                            except:
                                tfidf[docs] = doc_list[i][idx][1]
            end = time.time()

        elif len(doc_list) == 4:
            tfidf={}
            intersect = list(set(Extract(doc_list[0])) & set(Extract(doc_list[1])) & set(Extract(doc_list[2])) & set(Extract(doc_list[3])))
            for docs in intersect:
                for i in range(0, len(doc_list)):
                    for item in doc_list[i]:
                        #print(item)
                        exists = bool(docs in item)
                        if exists:
                            try:
                                idx = doc_list[i].index(item)
                                # print(idx)
                                tfidf[docs] += float(doc_list[i][idx][1])
                            except:
                                tfidf[docs] = doc_list[i][idx][1]
        else:
            continue

        if len(tfidf) !=0:
            #print("Word set: {}\nIntersection: {}".format(w_set, intersect))
           # print(tfidf)
            tfidf =dict(sorted(tfidf.items(), key=lambda itemm: itemm[1], reverse=True))
            # print(tfidf.keys())
            #for item in tfidf.keys():
                #print("\t{}".format(item))
                #x = collection.find_one({'_id': item}, {"Path": 0, 'Article': 0})
                #print(x) # UNCOMMENT
            #print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

            #print('\n')
        else:
            print("Word set: {}".format(w_set))
            print("No articles containing these words exist.\n")
    end = time.time()
    print('Elapsed time for {}-word queries: {} ms'.format(len(w_set), (end - start)*1000 / len(query_set)))

