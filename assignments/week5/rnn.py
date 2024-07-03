# based on: https://jkk.name/neural-tagger-tutorial/
import random
import codecs
from torch import nn
import torch
import sys

torch.manual_seed(0)
PAD = "PAD"
DIM_EMBEDDING = 100
LSTM_HIDDEN = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 10

def read_data(file_name):
    """
    read in conll file
    
    :param file_name: path to read from
    :returns: list with sequences of words and labels for each sentence
    """
    data = []
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
                data.append((current_words, current_tags))
            current_words = []
            current_tags = []

    # check for last one
    if current_tags != [] and not raw:
        data.append((current_words, current_tags))
    return data


train_data=read_data(sys.argv[1])

# Create vocabularies for both the tokens
# and the tags
id_to_token = [PAD]
token_to_id = {PAD: 0}
id_to_tag = [PAD]
tag_to_id = {PAD: 0}

for tokens, tags in train_data:
    for token in tokens:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
            id_to_token.append(token)
    for tag in tags:
        if tag not in tag_to_id:
            tag_to_id[tag] = len(tag_to_id)
            id_to_tag.append(tag)
NWORDS = len(token_to_id)
NTAGS = len(tag_to_id)


max_len=max([len(x[0]) for x in train_data ])

# convert text data with labels to indices
def data2feats(inputData, word2idx, label2idx):
    feats = torch.zeros((len(inputData), max_len), dtype=torch.long)
    labels = torch.zeros((len(inputData), max_len), dtype=torch.long)
    for sentPos, sent in enumerate(inputData):
        for wordPos, word in enumerate(sent[0][:max_len]):
            wordIdx = word2idx[PAD] if word not in word2idx else word2idx[word]
            feats[sentPos][wordPos] = wordIdx
        for labelPos, label in enumerate(sent[1][:max_len]):
            labelIdx = word2idx[PAD] if label not in label2idx else label2idx[label]
            labels[sentPos][labelPos] = labelIdx
    return feats, labels

train_feats, train_labels = data2feats(train_data, token_to_id, tag_to_id)


# convert to batches
num_batches = int(len(train_feats)/BATCH_SIZE)
train_feats_batches = train_feats[:BATCH_SIZE*num_batches].view(num_batches, BATCH_SIZE, max_len)
train_labels_batches = train_labels[:BATCH_SIZE*num_batches].view(num_batches, BATCH_SIZE, max_len)

class TaggerModel(torch.nn.Module):
    def __init__(self, nwords, ntags):
        super().__init__()

        # Create word embeddings
        self.word_embedding = nn.Embedding(nwords, DIM_EMBEDDING)
        # Create input dropout parameter
        self.word_dropout = torch.nn.Dropout(.2)
        # Create LSTM parameters
        self.rnn = torch.nn.RNN(DIM_EMBEDDING, LSTM_HIDDEN, num_layers=1,
                batch_first=True, bidirectional=False)
        # Create output dropout parameter
        self.rnn_output_dropout = torch.nn.Dropout(.3)
        # Create final matrix multiply parameters
        self.hidden_to_tag = torch.nn.Linear(LSTM_HIDDEN, ntags)

    def forward(self, sentences):
        # Look up word vectors
        word_vectors = self.word_embedding(sentences)
        # Apply dropout
        dropped_word_vectors = self.word_dropout(word_vectors)
        rnn_out, _ = self.rnn(dropped_word_vectors, None)
        # Apply dropout
        rnn_out_dropped = self.rnn_output_dropout(rnn_out)
        # Matrix multiply to get scores for each tag
        output_scores = self.hidden_to_tag(rnn_out_dropped)

        # Calculate loss and predictions
        return output_scores


model = TaggerModel(NWORDS, NTAGS)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

for epoch in range(EPOCHS):
    model.train() 
    model.zero_grad()

    # Loop over batches
    loss = 0
    match = 0
    total = 0
    for batchIdx in range(0, num_batches):
        output_scores = model.forward(train_feats_batches[batchIdx])
        output_scores = output_scores.view(BATCH_SIZE * max_len, -1)
        flat_labels = train_labels_batches[batchIdx].view(BATCH_SIZE * max_len)
        batch_loss = loss_function(output_scores, flat_labels)

        predicted_tags  = torch.argmax(output_scores, 1)
        predicted_tags = predicted_tags.view(BATCH_SIZE, max_len)

        # Prepare inputs
        input_array = train_feats_batches[batchIdx]
        output_array = train_labels_batches[batchIdx]

        # Construct computation
        output_scores = model(input_array)
        # Calculate loss
        output_scores = output_scores.view(BATCH_SIZE * max_len, -1)
        flat_labels = output_array.view(BATCH_SIZE * max_len)
        batch_loss = loss_function(output_scores, flat_labels)

        # Run computations
        batch_loss.backward()
        optimizer.step()
        model.zero_grad()
        loss += batch_loss.item()
        # Update the number of correct tags and total tags
        for goldSent, predSent in zip(train_labels_batches[batchIdx], predicted_tags):
            for goldLabel, predLabel in zip(goldSent, predSent):
                if goldLabel != 0:
                    total += 1
                    if goldLabel == predLabel:
                        match+= 1
    print(epoch, loss, match / total)


def run_eval(feats_batches, labels_batches):
    model.eval()
    match = 0
    total = 0
    for sents, labels in zip(feats_batches, labels_batches):
        output_scores = model.forward(sents)
        predicted_tags  = torch.argmax(output_scores, 2)
        for goldSent, predSent in zip(labels, predicted_tags):
            for goldLabel, predLabel in zip(goldSent, predSent):
                if goldLabel.item() != 0:
                    total += 1
                    if goldLabel.item() == predLabel.item():
                        match+= 1
    return(match/total)


for devPath in sys.argv[2:]:
    BATCH_SIZE=1
    dev_data=read_data(devPath)
    dev_feats, dev_labels = data2feats(dev_data, token_to_id, tag_to_id)
    num_batches2 = int(len(dev_feats)/BATCH_SIZE)

    dev_feats_batches = dev_feats[:BATCH_SIZE*num_batches2].view(num_batches2, BATCH_SIZE, max_len)
    dev_labels_batches = dev_labels[:BATCH_SIZE*num_batches2].view(num_batches2, BATCH_SIZE, max_len)
    score = run_eval(dev_feats_batches, dev_labels_batches)
    print(devPath, score)

