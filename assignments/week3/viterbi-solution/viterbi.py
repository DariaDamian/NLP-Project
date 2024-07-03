import pickle
import pprint



BEG = '<S>'
END = '</S>'
UNK = '<UNK>'


def viterbi(sent):
    emission_probs, transition_probs = pickle.load(open('probs_en.pickle', 'rb'))

    labels = sorted([x for x in emission_probs])
    print(labels)

    # scores: #words #labels
    scores = []
    backpointers = []
    for word in sent:
        scores.append([])
        backpointers.append([])
        for label in labels:
            scores[-1].append(0.0)
            backpointers[-1].append(0)

    #First position
    for labelIdx, label in enumerate(labels):
        if labelIdx < 2:
            continue
        word = sentence[0]
        if word in emission_probs[label]:
            emission = emission_probs[label][word]
        else:
            emission = emission_probs[label][UNK]
        transition = transition_probs[BEG][label]
        scores[0][labelIdx] = emission * transition

    for wordIdx in range(1, len(sentence)):
        word = sentence[wordIdx]
        # Get emission, because it is the same for each label

        for labelIdx, label in enumerate(labels):    
            if labelIdx < 2:
                continue

            if word in emission_probs[label]:
                emission = emission_probs[label][word]
            else:
                emission = emission_probs[label][UNK]

            # prev labels
            prevPathScores = [] 
            for prevLabelIdx, prevLabel in enumerate(labels):
                if prevLabelIdx < 2:
                    prevPathScores.append(0.0)
                    continue
                transition = transition_probs[prevLabel][label]
                history = scores[wordIdx-1][prevLabelIdx]
                prevPathScores.append(emission * transition * history)
            bestScore = max(prevPathScores)
            scores[wordIdx][labelIdx] = bestScore

            bestPrev = prevPathScores.index(bestScore)
            backpointers[wordIdx][labelIdx] = bestPrev


    # last step:
    prevPathScores = []
    for prevLabelIdx, prevLabel in enumerate(labels):
        if prevLabelIdx < 2:
            prevPathScores.append(0.0)
            continue
        transition = transition_probs[prevLabel][END]
        history = scores[-1][prevLabelIdx]
        score = transition * history
        print(prevLabelIdx, score)
        prevPathScores.append(score)

    bestScore = max(prevPathScores)
    bestLast = prevPathScores.index(bestScore)
    sequence = [bestLast]
    for wordIdx in list(reversed(range(1, len(sentence)))):
        bestNew = backpointers[wordIdx][bestLast]
        sequence.append(bestNew)
        bestLast = bestNew
    return list(reversed([labels[x] for x in sequence]))


sentence = ['this', 'is', 'fwsfvf', '.']
print(viterbi(sentence))

