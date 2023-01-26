import json
import os
import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
from pprint import pprint
import math
import re
from pymongo import MongoClient
import pickle
import en_core_web_sm
from nltk.corpus import wordnet
#nltk.download('averaged_perceptron_tagger')

nlp = en_core_web_sm.load()


# ~~~~~~~~~START: DATABASE CONNECTION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
client = MongoClient("mongodb+srv://athinafus:hannahm0ng0@lingtech.bv1wl.mongodb.net/Lingtech?retryWrites=true&w=majority")
db = client['Article_Data']
collection = db['Articles']
collection2 = db['Lemma']
# ~~~~~~~~~END: DATABASE CONNECTION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~START: FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def remove_punct(text):
    text.replace('"', '')
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)

def remove_stopwords(text, bad_tags):
    for word in text[:]:
        if word[1] in bad_tags:
            text.remove(word)
    return text


def lemmatize(text):
    lemmed = []
    l = [(w, get_wordnet_pos(w)[0]) for w in nltk.word_tokenize(text)]  # list that contains the PoS-tagged article
    l2 = [(w, get_wordnet_pos(w)[1]) for w in nltk.word_tokenize(text)]  # list that contains the FULL PoStagged article
    for el in l:
        if el[1] != 'stopw':
            ll = wordnet_lemmatizer.lemmatize(el[0], el[1])
            lemmed.append(ll)
    for tup in l2[:]:
        if tup[1] == 't':
            l2.remove(tup)

    return  (lemmed, l2)


def calculate_freq(word_list):
    temp_list = [0]*len(word_list)
    for i in range(0, len(word_list)):
        freq = word_list.count(word_list[i])
        temp_list[i] = (word_list[i], freq)
    return temp_list


def remove_emoji(string):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", string)


def clean_article(text):
    people_names = []
    doc = nlp(text)
    if len(doc.ents) != 0:  # if there are named entities in the text
        # --SYNTAX---[f(x) for x in sequence if condition]--------------------------------------------------------------
        [people_names.append(X.text) for X in doc.ents
         if X.label_ in ['QUANTITY', 'DATE', 'TIME', 'PERCENT', 'CARDINAL']]
    else:
        print('No named entities in text')
    name_regex = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, reversed(people_names))))
    text = name_regex.sub("", text)  # removing the named entities from the text
    text = text.replace("_", ' ')
    text = text.replace("-", ' ')
    text = text.replace("’", ' ')
    text = ''.join([i for i in text if not i.isdigit()])  # removing numbers from the text
    text = re.sub(' +', ' ', text)  # removing too many spaces from text

    return text



def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tagg = nltk.pos_tag([word])
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    #print(tagg[0][1])
    closed_class_category_tags = ['CC', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP',
                                  'TO', 'UH', 'WDT', 'WP', 'WP$', 'WRB']
    if tagg[0][1]  in closed_class_category_tags: # detecting stopwords
        return ("stopw", tagg[0][1])
    else:
        return (tag_dict.get(tag, wordnet.NOUN), tagg[0][1])

def remove_one_lengthed(article_list):
    [article_list.remove(x) for x in article_list[:] if len(x) == 1]
    return article_list

def lower_letters(word_list):
    word_list = [word.lower() for word in word_list]
    return word_list


# ~~~~~~~~~~END: FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

file_list=[]
df = pd.DataFrame(columns=['Document', 'Url', 'Path', 'Title', 'Article'])

for file in os.listdir("/home/athinafus/Documents/LinguisticTech/lingtech"):
    if file.endswith(".json"):
        file_list.append("{}".format(os.path.join("/home/athinafus/Documents/LinguisticTech/lingtech", file)))

counter = 1
for item in file_list:
    print(item)
    with open(item) as f:
        #data = json.load(f)
        list_of_jsons = f.readlines()
        #print(type(list_of_jsons))
        #print(list_of_jsons[2])
        for article in list_of_jsons[1:]:
            #print(article)
            try:
                response = article.replace('\n', '')
                response = response.replace('}{', '},{')
                response = response.replace('},', '}')
                #response = "[" + response + "]"
                #print(response)
                y = json.loads(response)
                #print(y)
                df = df.append({'Document': 'd{}'.format(counter), 'Url': y['url'], 'Path': str(item), 'Title': y['title'], 'Article': str(y['text'])},
                               ignore_index=True) # article in dataframe without any processing
                counter += 1
            except:
                continue


lengthBefore = len(df)
df.drop_duplicates(subset=['Title'], inplace=True, keep="first") # dropping duplicate articles
# length after removing duplicates
lengthAfter = len(df)
print(lengthAfter)
print(df.head(10).to_string())

# inserting the articles in the DATABASE without the duplicates
'''try:
    for index, row in df.iterrows():
        collection.insert_one({'_id': row['Document'], 'Url': row["Url"], 'Path': row["Path"], 'Title': row['Title'],
                               'Article': row['Article']})
except:
    print("PROBLEM OCCURRED DURING THE INSERTION OF ARTICLES TO THE DATABASE")'''


print('\n\n')

df["Article"] = [clean_article(i) for i in df['Article']] # cleaning article
df["Article"] = [remove_punct(i) for i in df['Article']]  # removing punctuation from the article
df["Article"] = [remove_emoji(i) for i in df['Article']]  # removing emojis from the article

#POS-Tagging
df_posT = df.copy() #new dataframe that contains the words and their assigned POS-tags

wordnet_lemmatizer = WordNetLemmatizer()
df_posT['Article'] = df_posT['Article'].apply(lambda x: lemmatize(x)[1]) # dataframe with tagged words to save in csv
df_posT.to_csv('pos_tagged_articles.csv', encoding='utf-8' )

df['Article'] = df['Article'].apply(lambda x: lemmatize(x)[0]) # lemmatize each word
df['Article'] = df['Article'].apply(lambda x: remove_one_lengthed(x))  # remove words of one character
df['Article'] = df['Article'].apply(lambda x: lower_letters(x)) # make all words lowercase
df['Article'] = df['Article'].apply(lambda x: calculate_freq(x)) # count frequency of each lemma in a document
df['Article'] = df['Article'].apply(lambda x: list(set(x))) # remove duplicate words


dict_index = {}
#creating the inverted file
for index, row in df.iterrows():
    for tup in row['Article']:
        try:
            dict_index[tup[0]].append([row['Document'], tup[1]])
        except:
            dict_index[tup[0]] = [[row['Document'], tup[1]]]

dict_index = dict(sorted(dict_index.items(), key=lambda itemm: itemm[1]))

list_for_up=[]
for item in dict_index:
    term_idf = int(lengthAfter)/int(len(dict_index[item]))
    for i in range(0, len(dict_index[item])):
        #element = [doc_id, freq]
        dict_index[item][i][1] = (1 + math.log(int(dict_index[item][i][1]),2))*term_idf
        # inserting the lemmas to the database
        list_for_up.append(
            {'Lemma': item, 'inDocument': dict_index[item][i][0], 'tfidf': dict_index[item][i][1]})

collection2.insert_many(list_for_up) # uploading the inverted file information to the database

with open('output2_20_jan.txt', 'wt') as out:
    pprint(dict_index, stream=out)

# saving dictionary in pickle file
a_file = open("inversed_file20_jan.pkl", "wb")
pickle.dump(dict_index, a_file)
a_file.close()

# building the xml string for the inverted file
xml_string = "<inverted_index>\n"
for item in dict_index:
    xml_string+='<lemma name="{}">\n'.format(item)
    for lists in dict_index[item]:
        xml_string+='<document id="{}" weight="{}"/>\n'.format(lists[0], lists[1])
    xml_string+='</lemma>\n'

xml_string += "</>\n"
f = open("inversed_file_20_jan.xml", "w") # xml file with inverted file
f.write(xml_string)
f.close()
