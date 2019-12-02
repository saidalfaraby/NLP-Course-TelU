# Ade Romadhony
# Fakultas Informatika, Telkom University
# sumber: http://nbviewer.ipython.org/github/tpeng/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb

from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite

train_lines = []
test_lines = []
with open('kalimat_POS_NE_train.txt', 'r') as f:
    train_lines = f.readlines()
with open('kalimat_POS_NE_test.txt', 'r') as f:
    test_lines = f.readlines()

train_sents = []
test_sents = []
sent = []
counter = 0
for line in train_lines:
    line = line.rstrip('\n')
    curr_tuple = ()
    if len(line)>1:
        line_part = line.split(" ")
        t = (line_part[0], line_part[1], line_part[2])
        #print(t)
        sent.append(t)
    else:
        print("train sent = ")
        print(sent)
        train_sents.append(sent)
        sent = []
        counter = counter+1
counter = 0
for line in test_lines:
    line = line.rstrip('\n')
    curr_tuple = ()
    if len(line)>1:
        line_part = line.split(" ")
        t = (line_part[0], line_part[1], line_part[2])
        #print(t)
        sent.append(t)
    else:
        print("test sent = ")
        print(sent)
        test_sents.append(sent)
        sent = []
        counter = counter+1




def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]  

print(sent2features(train_sents[0])[0])

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.train('sample.crfsuite')

tagger = pycrfsuite.Tagger()
tagger.open('sample.crfsuite')

example_sent = test_sents[0]
print(' '.join(sent2tokens(example_sent)), end='\n\n')

print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
print("Correct:  ", ' '.join(sent2labels(example_sent)))
