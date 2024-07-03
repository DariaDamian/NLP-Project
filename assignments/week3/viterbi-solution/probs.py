emission_counts = {}
for cur_label in label_list:
    emission_counts[cur_label] = {}
    
transition_counts = {}
for prev_label in label_list:
    transition_counts[prev_label] = {}
    for cur_label in label_list:
        transition_counts[prev_label][cur_label] = 0
    

for words, labels in read_conll_file('pos-data/train-students-da.conll'):
    for word, label in zip(words, labels):
        if word in emission_counts[label]:
            emission_counts[label][word] += 1
        else:
            emission_counts[label][word] = 1
    labels = [BEG] + labels + [END]
    for label_idx in range(1, len(labels)):
        prev_label = labels[label_idx-1]
        cur_label = labels[label_idx]
        transition_counts[prev_label][cur_label] += 1

print(emission_counts['ADJ'])
print(transition_counts['NOUN'])

# Add .1 to all tokens, and add UNK with .1 probability
for label in emission_counts:
    for word in emission_counts[label]:
        emission_counts[label][word] += .1
    emission_counts[label][UNK] = .1

for prev_label in transition_counts:
    for cur_label in transition_counts[prev_label]:
        transition_counts[prev_label][cur_label] += .1
    transition_counts[prev_label][UNK] = .1
    
# convert to probability
emission_probs = {}
for label in emission_counts:
    emission_probs[label] = {}
    total = sum([emission_counts[label][word] for word in emission_counts[label]])
    for word in emission_counts[label]:
        emission_probs[label][word] = emission_counts[label][word] / total
        
transition_probs = {}
for prev_label in transition_counts:
    transition_probs[prev_label] = {}
    total = sum([transition_counts[prev_label][cur_label] for cur_label in transition_counts[prev_label]])
    for cur_label in transition_counts[prev_label]:
        transition_probs[prev_label][cur_label] = transition_counts[prev_label][cur_label] / total
        
# Doing both steps at once would increase efficiency, but is less "debugable"
import pickle
pickleF = open('probs_da.pickle', 'wb')
pickle.dump([emission_probs, transition_probs], pickleF)
pickleF.close()
