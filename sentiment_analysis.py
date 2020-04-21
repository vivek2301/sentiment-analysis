import torch
from torchtext import data
from model import LSTM
import random
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

SEED = 1234
BATCH_SIZE = 64
N_EPOCHS = 10
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def main():
    #Loop over tokenizer options dictionary
    for tokenizerType in tokenizerOptions:
        classify(tokenizerType)

def classify(tokenizerType):
    #Load dataset
    
    TEXT = data.Field(tokenize = tokenizerOptions[tokenizerType], include_lengths = True, lower=True)
    LABEL = data.LabelField(dtype=torch.float, sequential=False, use_vocab=False)
    fields = [('text', TEXT), ('label', LABEL)]
    train_data = data.TabularDataset(
        path='SJ_Unsupervised_NLP_data.txt',
        format='tsv',
        fields=fields
    )

    #Split dataset into train, validation and test
    train_data, valid_data, test_data = train_data.split(
        split_ratio=[0.64, 0.2, 0.16],
        random_state=random.seed(SEED)
    )

    #Build vocabulary using predefined vectors
    TEXT.build_vocab(
        train_data, 
        vectors = "glove.6B.100d", 
        unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(train_data)

    #print(TEXT.vocab.itos[:100])
    #Use GPU, if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Create iterators to get data in batches
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        datasets=(train_data, valid_data, test_data),
        batch_size = BATCH_SIZE,
        device = device,
        sort_key=lambda x: len(x.text),
        sort=False,
        sort_within_batch=True
    )

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1

    model = LSTM(
        vocab_size = INPUT_DIM, 
        embedding_dim = EMBEDDING_DIM,
        hidden_dim = HIDDEN_DIM,
        output_dim = OUTPUT_DIM,
        n_layers = 3,
        bidirectional = True,
        dropout = 0.5,
        pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    )

    #Replace initial weights of embedding with pre-trained embedding
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    #Set UNK and PAD embeddings to zero
    model.embedding.weight.data[TEXT.vocab.stoi[TEXT.unk_token]] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[TEXT.vocab.stoi[TEXT.pad_token]] = torch.zeros(EMBEDDING_DIM)

    #SGD optimizer and binary cross entropy loss
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    #Transfer model and criterion to GPU
    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = float('inf')
    train_loss_list = []
    valid_loss_list = []

    for epoch in range(N_EPOCHS):
        
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best-model.pt')
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
    
    print(tokenizerType + ":")
    plotLoss(train_loss_list, valid_loss_list)

    model.load_state_dict(torch.load('best-model.pt'))

    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    print("\n")

#Plot training and validation loss
def plotLoss(train_loss, valid_loss):
    epochs = range(N_EPOCHS)
    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, valid_loss, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

#Custom tokenizer
def tokenizer(text, exceptPOS): # create a tokenizer function
    if exceptPOS == None:
        return word_tokenize(text)
    token=[]
    for tok in nltk.pos_tag(word_tokenize(text)):
        if tok[1] in exceptPOS :
            continue
        token.append(tok[0])
    return token

def tokenizerExceptNouns(text):
    nouns = {'NN', 'NNS', 'NNP', 'NNPS'}
    return tokenizer(text.lower(), nouns)

def tokenizerExceptVerbs(text):
    verbs = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    return tokenizer(text.lower(), verbs)

def tokenizerExceptAdjectives(text):
    adjectives = {'JJ', 'JJR', 'JJS'}
    return tokenizer(text.lower(), adjectives)

def tokenizerWithAll(text):
    return tokenizer(text.lower(), None)

tokenizerOptions = {
    "Full_Data" : tokenizerWithAll,
    "No_Nouns" : tokenizerExceptNouns,
    "No_Verbs" : tokenizerExceptVerbs,
    "No_Adjectives" : tokenizerExceptAdjectives
}

#Function to train the model
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

#Forward pass for classification
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

if __name__ == '__main__':
    main()