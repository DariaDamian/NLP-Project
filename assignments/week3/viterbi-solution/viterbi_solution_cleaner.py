import codecs

def read_conll_file(file_name):
    """
    read in conll file
    
    :param file_name: path to read from
    :yields: list of words and labels for each sentence
    """
    current_words = []
    current_tags = []

    for line in codecs.open(file_name, encoding='utf-8'):
        line = line.strip()

        if line:
            if line[0] == '#':
                continue # skip comments
            tok = line.split('\t')
            word = tok[0]
            tag = tok[1]

            current_words.append(word)
            current_tags.append(tag)
        else:
            if current_words:  # skip empty lines
                yield((current_words, current_tags))
            current_words = []
            current_tags = []

    # check for last one
    if current_tags != [] and not raw:
        yield((current_words, current_tags))

label_set = set()
for words, labels in read_conll_file('pos-data/da_arto-train.conll'):
    for label in labels:
        label_set.add(label)
print(label_set)

SMOOTH = 0.1
UNK = '<UNK>'
BEG = '<S>'
END = '</S>'


# emission probs:
emissions = {} # final result, give label then word to find emission prob of word
totals = {} # total count for each label needed for C(ti) in formula

for label in label_set:
    # 1 for smoothing!
    emissions[label] = {UNK:SMOOTH}
    totals[label] = SMOOTH
    
for words, labels in read_conll_file('pos-data/da_arto-train.conll'):
    for word, label in zip(words, labels):
        totals[label] += 1 #Originally SMOOTH but not sure why
        if word not in emissions[label]:
            emissions[label][word] = 1 + SMOOTH # 2 because of smoothing!
        else:
            emissions[label][word] += 1

# got the counts, now turn them into probs
for label in emissions:
    for word in emissions[label]:
        emissions[label][word] /= totals[label]

# to deal with UNK
def emissionProb(label,word):
    if word in emissions[label].keys():
        return emissions[label][word]
    else:
        return emissions[label][UNK]

# transmission prob:

tagCounts = {} #Counts of next tags for each tag

label_set_ext = label_set.copy() #label_set defined in first code cell
label_set_ext.add(BEG) 
label_set_ext.add(END)

# Smoothing
for label in label_set_ext:
    tagCounts[label] = {}
    for label2 in label_set_ext:
        tagCounts[label].setdefault(label2,SMOOTH)

for _, labels in read_conll_file('pos-data/da_arto-train.conll'):
    for labelIdx in range(len(labels)):
        
        curLabel = labels[labelIdx]
        if labelIdx == 0: # Start of sentence is handled differently
            prev = BEG
        else:
            prev = labels[labelIdx-1]
        
        tagCounts[prev][curLabel] += 1
        
    # add prob. to </S> i.e end of sentence is handled differently
    tagCounts[curLabel][END] += 1

# Summing counts for each tag to get tag priors
tagCountSums = {tag:sum(tagCounts[tag].values()) for tag in tagCounts.keys()}

for tag1 in tagCounts:
    for tag2 in tagCounts[tag1]:
        tagCounts[tag1][tag2] /= tagCountSums[tag1]
        
transition = tagCounts
print(transition['ADJ'])

import numpy as np
def viterbi(sentence):
    row_count = len(label_set)
    labels = list(label_set)

    # scores is of shape(labels,words)
    scores = np.array([[0.0]*len(sentence) for i in range(row_count)])
    came_from = np.array([[0]*len(sentence) for i in range(row_count)])
    
    for idx, word in enumerate(sentence):
        for jdx, tag in enumerate(labels):
            if tag in [BEG,END]:
                continue
            
            if idx == 0:
                for x in range(len(labels)):
                    scores[jdx,idx] = emissionProb(tag,word)*transition[BEG][tag]
            
            else:
                cand_scores = [0]*len(labels)
                for kdx, candlabel in enumerate(labels):
                    #print(tag,candlabel)
                    cand_scores[kdx] = emissionProb(tag,word)*transition[candlabel][tag]*scores[kdx,idx-1]
                scores[jdx,idx] = max(cand_scores)
                came_from[jdx,idx] = np.argmax(cand_scores)
            
    path = [np.argmax(scores[:,-1])]
    for i in range(len(sentence)-1,0,-1):
        node = path[-1]
        path.append(int(came_from[int(node),i]))
    path.reverse()
    res_labels = []
    for i in path:
        res_labels.append(labels[i])
    return res_labels
        

#analysis and accuracy
total = 0
correct = 0
confusions= {}
data = list(read_conll_file('pos-data/da_arto-dev.conll'))
for words, labels in data:
    res_labels = viterbi(words)
    for x,y in zip(labels, res_labels):
        total += 1
        if x == y:
            correct += 1
        else:
            confusion = x + '-' + y
            if confusion in confusions:
                confusions[confusion] += 1
            else:
                confusions[confusion] = 1
print(correct/total)  
for k in sorted(confusions, key=confusions.get, reverse=True)[:10]:
    print(k, confusions[k])

