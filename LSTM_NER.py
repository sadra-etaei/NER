import torch
import numpy as np
from torch import nn
from data_utils import lstm_collate_fn,custom_data_loader_multiclass
from torch.utils.data import DataLoader
from functools import partial
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

print(f"Using execution accelerator target: {device}")

train_sentences,train_labels,val_sentences,val_labels,vocabulary,word2id = custom_data_loader_multiclass("D:/projects/nlp/NER/conll2003/eng.train","D:/projects/nlp/NER/conll2003/eng.testa")




class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
        super().__init__()
        self.pad_id =pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim,num_layers=4, batch_first=True, bidirectional=True,dropout=0.5)
  
        self.fc = nn.Linear(hidden_dim * 2, num_classes) 
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded) # Shape: (B, L, hidden_dim * 2)
        logits = self.fc(lstm_out)        # Shape: (B, L, num_classes)
        return logits
    




def multiclass_loss_func(batch_outputs, batch_labels):
    B, L, num_classes = batch_outputs.size()
    
    outputs_flat = batch_outputs.view(B * L, num_classes)
    labels_flat = batch_labels.view(B * L)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    return criterion(outputs_flat, labels_flat)


def train_multiclass(model, loss_func, optimizer, data, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_inputs, batch_labels, batch_lengths in data:
            optimizer.zero_grad()

            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            
            logits = model(batch_inputs)
            
            loss = loss_func(logits, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        epoch_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {epoch_loss:.4f}")

def evaluate_multiclass(model, val_data):
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for batch_inputs, batch_labels, batch_lengths in val_data:

            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_inputs)
            
            predictions = torch.argmax(logits, dim=-1)
            
            preds_flat = predictions.cpu().numpy().flatten()
            labels_flat = batch_labels.cpu().numpy().flatten()
            
            valid_indices = (labels_flat != -100)
            
            all_predictions.extend(preds_flat[valid_indices])
            all_ground_truths.extend(labels_flat[valid_indices])
            
    all_ground_truths = np.array(all_ground_truths)
    all_predictions = np.array(all_predictions)
    true_entity_mask = (all_ground_truths != 0)
    total_true_entities = np.sum(true_entity_mask)
    
    correct_guesses = np.sum((all_predictions == all_ground_truths) & true_entity_mask)
    
    false_positives = np.sum((all_predictions != 0) & (all_ground_truths == 0))
    
    entity_classes = [1, 2, 3, 4, 5, 6, 7, 8]
    
    precision = precision_score(all_ground_truths, all_predictions, labels=entity_classes, average='macro', zero_division=0)
    recall = recall_score(all_ground_truths, all_predictions, labels=entity_classes, average='macro', zero_division=0)
    f1 = f1_score(all_ground_truths, all_predictions, labels=entity_classes, average='macro', zero_division=0)
    
    print("\n================ DETAILED MULTICLASS METRICS ================")
    print(f"Total True Entity Tokens in Val Set : {total_true_entities}")
    print(f"Correctly Predicted Entity Tokens   : {correct_guesses}")
    print(f"False Positives (Spurious Alarms)   : {false_positives}")
    print("-------------------------------------------------------------")
    print(f"Macro Precision : {precision:.4f}  (When it finds an entity, how often is it right?)")
    print(f"Macro Recall    : {recall:.4f}  (What % of the actual entities did it catch?)")
    print(f"Macro F1-Score  : {f1:.4f}  (Overall structural performance balance)")
    print("=============================================================")
    
    model.train()

data = DataLoader(
    list(zip(train_sentences, train_labels)),
    batch_size=8,
    pin_memory=True,
    shuffle=True,
    collate_fn=partial(lstm_collate_fn, word2id=word2id) 
)

val_loader = DataLoader(
    list(zip(val_sentences,val_labels)),
    batch_size=8,
    shuffle=False,
    collate_fn=partial(lstm_collate_fn, word2id=word2id) 
)


model = BiLSTM_NER(len(word2id),128,128,9,word2id["<pad>"])

model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(),0.001)


train_multiclass(model,multiclass_loss_func,optimizer,data,10)



evaluate_multiclass(model,val_loader)



def predict_custom_sentence(sentence, model, word2id):
    """
    Takes a raw string sentence, runs it through the trained LSTM model,
    and prints out a word-by-word entity classification alignment report.
    """
    model.eval()
    
    target_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    
    raw_words = sentence.strip().split()
    processed_words = [w.lower() for w in raw_words] 
    
    unk_id = word2id.get("<unk>")
    word_ids = [word2id.get(w, unk_id) for w in processed_words]
    
  
    input_tensor = torch.LongTensor(word_ids).unsqueeze(0)
    
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        logits = model(input_tensor) 
        
        predictions = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy() # shape: [SequenceLength]
        
    
    print(f"\nInference Report for: \"{sentence}\"")
    print(f"{'Word':<18} | {'Predicted Class ID':<18} | {'Entity Tag'}")
    print("-" * 55)
    
    for word, class_id in zip(raw_words, predictions):
        tag_name = target_names[class_id]
        print(f"{word:<18} | {class_id:<18} | {tag_name}")
    print("=" * 55)
    
    model.train()




sample_sentence_1 = "George Washington traveled to Paris last week"
predict_custom_sentence(sample_sentence_1, model, word2id)

sample_sentence_2 = "The members of United Nations met in New York"
predict_custom_sentence(sample_sentence_2, model, word2id)