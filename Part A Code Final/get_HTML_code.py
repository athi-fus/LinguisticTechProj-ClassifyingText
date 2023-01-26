import requests
from pymongo import MongoClient


r = requests.get("https://example.com")
#print(r.text)

client = MongoClient("mongodb+srv://athinafus:hannahm0ng0@lingtech.bv1wl.mongodb.net/Lingtech?retryWrites=true&w=majority")
db = client['Article_Data']
collection = db['Articles']

#collection.delete_many({}) # deletes all entries

myresult = collection.find({},{ "Path": 0 , 'Article': 0}).limit(500)

for x in myresult:
    print(x['Url'])
    r = requests.get(x['Url'])
    f = open("/home/athinafus/Documents/LinguisticTech/lingtech/lingtech/html_articles/{}.html"
             .format(x['Title'].replace(' ', '_')), "w")
    f.write(r.text)
    f.close()

