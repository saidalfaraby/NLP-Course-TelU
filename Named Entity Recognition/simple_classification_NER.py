# Ade Romadhony
# Fakultas Informatika, Telkom University

from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

# baca file train dan test
train_lines = []
test_lines = []
with open('kalimat_POS_NE_train.txt', 'r') as f:
    train_lines = f.readlines()
with open('kalimat_POS_NE_test.txt', 'r') as f:
    test_lines = f.readlines()

# fungsi untuk konversi label NE
def convert_label(raw_ne):
    new_label = 0
    if (raw_ne=='B-PER'): new_label = 1
    if (raw_ne=='I-PER'): new_label = 2
    if (raw_ne=='B-ORG'): new_label = 3
    if (raw_ne=='I-ORG'): new_label = 4
    if (raw_ne=='B-LOC'): new_label = 5
    if (raw_ne=='I-LOC'): new_label = 6
    return new_label

# inisialisasi kode token dan postag
token_dict = {}
postag_dict = {}
counter_token = 0
counter_postag = 0

# baca data token, postag, dan label NE
train_sents = []
test_sents = []
sent = []
counter = 0
# data train
for line in train_lines:
    line = line.rstrip('\n')
    curr_tuple = ()
    if len(line)>1:
        line_part = line.split(" ")
        t = (line_part[0], line_part[1], convert_label(line_part[2]))
        if line_part[0].lower() not in token_dict.keys():
            token_dict[line_part[0].lower()] = counter_token
            counter_token = counter_token+1
        if line_part[1] not in postag_dict.keys():
            postag_dict[line_part[1]] = counter_postag
            counter_postag = counter_postag+1
        #print(t)
        sent.append(t)
    else:
        print("train sent = ")
        print(sent)
        train_sents.append(sent)
        sent = []
        counter = counter+1

# data test
counter = 0
for line in test_lines:
    line = line.rstrip('\n')
    curr_tuple = ()
    if len(line)>1:
        line_part = line.split(" ")
        t = (line_part[0], line_part[1], convert_label(line_part[2]))
        #print(t)
        sent.append(t)
    else:
        print("test sent = ")
        print(sent)
        test_sents.append(sent)
        sent = []
        counter = counter+1


# kode untuk token/kata dan postag yang tidak muncul di data training, namun muncul di data testing
token_dict['unk'] = 9999
postag_dict['unk'] = 9999

# fungsi untuk ekstraksi fitur dari sebuah kalimat
def word2features(sent, i):  
    word = sent[i][0]
    postag = sent[i][1]
    if word.lower() not in token_dict.keys(): 
        word = 'unk'
    if postag not in postag_dict.keys():
        postag = 'unk'
    features = [
        token_dict[word.lower()], # fitur kata dalam bentuk lowercase
        word.isupper(), # fitur apakah karakter pertama token merupakan huruf kapital
        word.istitle(), # fitur apakah token merupakan title
        word.isdigit(), # fitur apakah token merupakan digit
        postag_dict[postag] # fitur kode postag token
    ]
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]  

print('postag dictonary')
print(postag_dict)
print('token dictonary')
print(token_dict)

# ekstraksi fitur data train
X_train = []
y_train = []
for s in train_sents:
    for i in range(len(s)):
        X_train.append(word2features(s,i))
        y_train.append(s[i][2])
# ekstraksi fitur data test
X_test = []
y_test = []
for s in test_sents:
    for i in range(len(s)):
        X_test.append(word2features(s,i))
        y_test.append(s[i][2])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

# train classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Coba test, satu kata
print('xtest')
print(X_test[0])

print(clf.predict(X_test[0].reshape(1,-1)))

# Coba test, keseluruhan data test
print('hasil klasifikasi data test:')
print(clf.predict(X_test))
