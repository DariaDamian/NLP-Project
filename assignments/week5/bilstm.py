import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F


def load_topics(path):
    text = []
    labels = []
    for lineIdx, line in enumerate(open(path)):
        tok = line.strip().split('\t')
        labels.append(tok[0])
        text.append(tok[1].split(' '))
    return text, labels

topic_train_text, topic_train_labels = load_topics('topic-data/train.txt')
topic_dev_text, topic_dev_labels = load_topics('topic-data/dev.txt')

max_len=32
PAD = '<PAD>'

word2idx = {PAD:0}
idx2word = [PAD]

label2idx = {PAD:0, 'starwars':1, 'poke': 2, 'muppet': 3}
idx2label = [PAD, 'starwars', 'poke', 'muppet']

# generate word2idxs
for sentPos, sent in enumerate(topic_train_text):
    for wordPos, word in enumerate(sent[:max_len]):
        if word not in word2idx:
            word2idx[word] = len(idx2word)
            idx2word.append(word)

# function to convert input to labels for use in torch
def data2feats(text2, labels2):
    feats = torch.zeros((len(text2), max_len), dtype=torch.long)
    labels = torch.zeros((len(text2)), dtype=torch.long)
    for sentPos, sent in enumerate(text2):
        for wordPos, word in enumerate(sent[:max_len]):
            wordIdx = word2idx[PAD] if word not in word2idx else word2idx[word]
            feats[sentPos][wordPos] = wordIdx
        labels[sentPos] = label2idx[labels2[sentPos]]
    return feats, labels

train_feats, train_labels = data2feats(topic_train_text, topic_train_labels)
dev_feats, dev_labels = data2feats(topic_dev_text, topic_dev_labels)


batch_size = 64
num_batches = int(len(train_labels)/batch_size)
train_feats_batches = train_feats[:batch_size*num_batches].view(num_batches,batch_size, max_len)
train_labels_batches = train_labels[:batch_size*num_batches].view(num_batches, batch_size)


lstm_dim = 50
embed_dim = 100

class LangID(nn.Module):
    def __init__(self, embed_dim, lstm_dim, vocab_dim):
        super(LangID, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_dim, embed_dim)
        self.word_dropout = nn.Dropout(.2)
        self.lstm = torch.nn.LSTM(embed_dim, lstm_dim, bidirectional=True, batch_first=True)
        self.output_dropout = nn.Dropout(.3)
        self.hidden_to_tag = torch.nn.Linear(lstm_dim * 2, len(idx2label))
        
    
    def forward(self, inputs):
        word_vectors = self.word_embeddings(inputs)
        #word_vectors =  self.word_dropout(word_vectors)
        lstm_out, _ = self.lstm(word_vectors)
        #lstm_out = self.output_dropout(lstm_out)
        y = self.hidden_to_tag(lstm_out[:,-1,:]) # NOTE THE [-1], this is because we only have 1 output label
        log_probs = F.softmax(y)
        return log_probs
    
    def predict(self, inputs):
        with torch.no_grad():
            y = self.forward(inputs)
            outputs = []
            for output in y:
                outputs.append(output.tolist().index(max(output.tolist())))
        return outputs


langid_model = LangID(embed_dim,lstm_dim, len(idx2word))
loss_function = nn.CrossEntropyLoss()

# compile and train the model
optimizer = optim.Adam(langid_model.parameters(), lr=0.01)
start = time.time()

langid_model.train()
for epoch in range(5):
    epoch_loss = 0.0
    for feats, label in zip(train_feats_batches, train_labels_batches):
        optimizer.zero_grad()
        #feats = feats.view(1,-1)
        #label = label.view(1,-1)
        y = langid_model.forward(feats)
        loss = loss_function(y,label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(epoch, epoch_loss/len(train_feats_batches), time.time() - start)
print(langid_model)


langid_model.eval()
cor = 0
for instanceIdx in range(len(dev_labels)):
    instanceFeats= dev_feats[instanceIdx]
    label = langid_model.predict(instanceFeats.view(1,32))[0]
    if label == dev_labels[instanceIdx].item():
        cor+=1
print(cor/len(dev_labels))



