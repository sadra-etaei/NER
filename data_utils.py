import torch
from torch import nn


def parse_conll_to_corpus(file_path):
    corpus_sentences = []
    corpus_labels = []
    
    current_words = []
    current_labels = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            if not line:
                if current_words:
                    corpus_sentences.append(current_words)
                    corpus_labels.append(current_labels)
                    current_words = []
                    current_labels = []
                continue
                
            if line.startswith("-DOCSTART-"):
                continue
                
            parts = line.split()
            if len(parts) >= 4:
                word = parts[0] 
                ner_tag = parts[3]
                
                current_words.append(word)
                binary_label = 0 if ner_tag == "O" else 1
                current_labels.append(binary_label)
                
        if current_words:
            corpus_sentences.append(current_words)
            corpus_labels.append(current_labels)
            
    return corpus_sentences, corpus_labels


def parse_conll_multiclass(file_path):
    tag2id = {
        "O": 0,
        "B-PER": 1, "I-PER": 2,
        "B-ORG": 3, "I-ORG": 4,
        "B-LOC": 5, "I-LOC": 6,
        "B-MISC": 7, "I-MISC": 8
    }
    
    sentences = []
    labels = []
    
    current_words = []
    current_labels = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith("-DOCSTART-"):
                if current_words:
                    sentences.append(current_words)
                    labels.append(current_labels)
                    current_words = []
                    current_labels = []
                continue
                
            splits = line.split()
            word = splits[0]
            ner_tag = splits[-1] 
            
            label_id = tag2id.get(ner_tag, 0)
            
            current_words.append(word)
            current_labels.append(label_id)
            
        if current_words:
            sentences.append(current_words)
            labels.append(current_labels)
            
    return sentences, labels


def pad_window(sentence, window_size, pad_token="<pad>"):
  window = [pad_token] * window_size
  return window + sentence + window


def convert_tokens_to_indices(sentence, word2id):
  return [word2id.get(word, word2id["<unk>"]) for word in sentence]



def custom_collate_fn(batch, window_size, word2id):
  x, y = zip(*batch)
  x = [pad_window(s, window_size=window_size) for s in x]
  x = [convert_tokens_to_indices(s, word2id) for s in x]

  pad_token_id = word2id["<pad>"]
  x = [torch.LongTensor(x_i) for x_i in x]
  x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=pad_token_id)

  lengths = [len(label) for label in y]
  lengths = torch.LongTensor(lengths)
  y = [torch.LongTensor(y_i) for y_i in y]
  y_padded = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)

  return x_padded, y_padded, lengths




def lstm_collate_fn(batch, word2id):
    x, y = zip(*batch)
    
    x_indices = [convert_tokens_to_indices(s, word2id) for s in x]
    
    x_tensors = [torch.LongTensor(x_i) for x_i in x_indices]
    y_tensors = [torch.LongTensor(y_i) for y_i in y]
    
    lengths = torch.LongTensor([len(labels) for labels in y])
    
    pad_token_id = word2id["<pad>"]
    x_padded = nn.utils.rnn.pad_sequence(x_tensors, batch_first=True, padding_value=pad_token_id)
    
    y_padded = nn.utils.rnn.pad_sequence(y_tensors, batch_first=True, padding_value=-100)
    
    return x_padded, y_padded, lengths




def custom_data_loader(train_path,val_path):
    train_sentences, train_labels = parse_conll_to_corpus(train_path)
    val_sentences, val_labels = parse_conll_to_corpus(val_path)
    
    vocabulary = set(word for sentence in train_sentences for word in sentence)
    vocabulary.add("<unk>")
    vocabulary.add("<pad>")

    word2id = {word: i for i, word in enumerate(sorted(list(vocabulary)))}


    return train_sentences,train_labels,val_sentences,val_labels,vocabulary,word2id


def custom_data_loader_multiclass(train_path,val_path):
    train_sentences, train_labels = parse_conll_multiclass(train_path)
    val_sentences, val_labels = parse_conll_multiclass(val_path)
    
    vocabulary = set(word for sentence in train_sentences for word in sentence)
    vocabulary.add("<unk>")
    vocabulary.add("<pad>")

    word2id = {word: i for i, word in enumerate(sorted(list(vocabulary)))}


    return train_sentences,train_labels,val_sentences,val_labels,vocabulary,word2id