import pickle
from pymongo import MongoClient
from operator import itemgetter
from nltk.stem import WordNetLemmatizer
import re
from prettytable import PrettyTable

def lemmatize(word_list):
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

wordnet_lemmatizer = WordNetLemmatizer()

while True:
    text = input("\nGive words to query. One, two, three or four. :")

    text = re.sub('[^0-9a-zA-Z]', ' ', text)  # no non-alphanumeric characters
    text = text.split(' ')
    text = [x.strip() for x in text]
    text = [x.lower() for x in text]
    text = list(filter(None, text)) # list of "clean" words
    text = lemmatize(text)


    doc_list = []
    myTable = PrettyTable(["Title", "Url"])
    for lem in text:

        try:
            # print(inverted_file[lem])
            doc_list.append(sorted(inverted_file[lem], key=lambda x: x[1], reverse=True))
            # sorted_tfidf = sorted(inverted_file[lem], key=lambda x: x[1], reverse=True)
        except Exception as e:
            print(e)
            print('LIST ERROR')
            # doc_list = sorted(inverted_file[lem], key=lambda x: x[1], reverse=True)
            # sorted_tfidf = sorted(doc_list, key=lambda x: x[1], reverse=True)

    if len(doc_list) == 2:
        tfidf = {}
        intersect = list(set(Extract(doc_list[0])) & set(Extract(doc_list[1])))
        for docs in intersect:
            for i in range(0, len(doc_list)):
                for item in doc_list[i]:
                    # print(docs)
                    # print(item)
                    exists = bool(docs in item)
                    if exists:
                        try:
                            idx = doc_list[i].index(item)
                            # print(idx)
                            tfidf[docs] += float(doc_list[i][idx][1])
                        except:
                            tfidf[docs] = doc_list[i][idx][1]




    elif len(doc_list) == 3:
        tfidf = {}
        intersect = list(set(Extract(doc_list[0])) & set(Extract(doc_list[1])) & set(Extract(doc_list[2])))
        for docs in intersect:
            for i in range(0, len(doc_list)):
                for item in doc_list[i]:
                    # print(item)
                    exists = bool(docs in item)
                    if exists:
                        try:
                            idx = doc_list[i].index(item)
                            # print(idx)
                            tfidf[docs] += float(doc_list[i][idx][1])
                        except:
                            tfidf[docs] = doc_list[i][idx][1]



    elif len(doc_list) == 4:
        tfidf = {}
        intersect = list(
            set(Extract(doc_list[0])) & set(Extract(doc_list[1])) & set(Extract(doc_list[2])) & set(
                Extract(doc_list[3])))
        for docs in intersect:
            for i in range(0, len(doc_list)):
                for item in doc_list[i]:
                    # print(item)
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

    if len(tfidf) != 0:
        # print("Word set: {}\nIntersection: {}".format(w_set, intersect))
        # print(tfidf)
        tfidf = dict(sorted(tfidf.items(), key=lambda itemm: itemm[1], reverse=True))
        # print(tfidf.keys())
        for item in tfidf.keys():
        # print("\t{}".format(item))
            x = collection.find_one({'_id': item}, {"Path": 0, 'Article': 0})
            #print(x) # UNCOMMENT
            myTable.add_row([x['Title'], x['Url']])
        # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print(myTable)

        # print('\n')
    else:
        print("Word set: {}".format(text))
        print("No articles containing these words exist.\n")

    ans = input("Would you like to make another search?\nType 'yes' or 'no':")
    ans = ans.lower()
    while(ans != 'yes') and (ans != 'no'):
        ans = input("Please type 'yes' or 'no': ")
    if ans == 'no':
        print("\n\nThank you for searching articles with us.\n\tGoodbye ^.^")
        break
