# Ade Romadhony
# Contoh sederhana NER berbasis aturan yang didefinisikan secara manual

# read the file
lines = []
with open('kalimat_POS_NE_train.txt', 'r') as f:
    lines = f.readlines()

counter_line = 0
tokens = []
postags = []
for line in lines:
    line = line.rstrip('\n')
    if len(line)>1:
        line_part = line.split(" ")
        tokens.append(line_part[0])
        postags.append(line_part[1])
    else:
        print(tokens)
        print(postags)
        NE_labels = []
        counter_token = 0
        prev_NE_label = ""
        for token in tokens:
            if token[0].isupper() and postags[counter_token]=='NNP':
                if prev_NE_label=="B-PER":
                    NE_labels.append("I-PER")
                else:
                    NE_labels.append("B-PER")
            else:
                NE_labels.append("O")
            prev_NE_label = NE_labels[counter_token]
            counter_token += 1
        print(tokens)
        print(NE_labels)
        tokens = []
        postags = []
