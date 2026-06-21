import torch
from torch import nn
from torch.utils.data import DataLoader
from functools import partial
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report
import numpy as np
from data_utils import custom_collate_fn,custom_data_loader,convert_tokens_to_indices


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

print(f"Using execution accelerator target: {device}")

train_sentences,train_labels,val_sentences,val_labels,vocabulary,word2id = custom_data_loader("D:/projects/nlp/NER/conll2003/eng.train","D:/projects/nlp/NER/conll2003/eng.testa")



data = DataLoader(list(zip(train_sentences,train_labels)),2,True,collate_fn=partial(custom_collate_fn,window_size=2,word2id=word2id))

class Named_Entity_Recognition(nn.Module):

  def __init__(self,vocab_size,batch_size,window_size,embed_dim,hidden_dim,pad_id=0):
    super(Named_Entity_Recognition,self).__init__()
    self.vocab_size = vocab_size
    self.batch_size = batch_size
    self.window_size = window_size
    self.embed_dim = embed_dim
    self.hidden_dim = hidden_dim
    self.pad_id = pad_id


    self.embed_layer = nn.Embedding(self.vocab_size,self.embed_dim,self.pad_id)


    self.full_window = 2 * window_size + 1

    self.hidden_layer = nn.Sequential(
      nn.Linear(self.full_window*self.embed_dim,self.hidden_dim),
      nn.Tanh()
    )


    self.output_layer = nn.Linear(self.hidden_dim,1)

    self.prob = nn.Sigmoid()


  def forward(self,inputs):
    B,L = inputs.size()


    token_windows = inputs.unfold(1,self.full_window,1)
    _,adjusted_length,_ = token_windows.size()


    assert token_windows.size() == (B, adjusted_length, self.full_window)

 
    embedding = self.embed_layer(token_windows)

    embedding = embedding.view(B,adjusted_length,-1)



    hidden = self.hidden_layer(embedding)

    output = self.output_layer(hidden)

    prob = self.prob(output)
    prob = prob.view(B,-1)
    

    return prob
  


model = Named_Entity_Recognition(len(word2id),2,2,25,25,word2id['<pad>'])


optimizer = torch.optim.Adam(model.parameters(),0.01)


def loss_func(batch_outputs, batch_labels, batch_lengths):
    max_output_len = batch_outputs.size(1)
    batch_labels_trimmed = batch_labels[:, :max_output_len]
    
    bce = nn.BCELoss()
    loss = bce(batch_outputs, batch_labels_trimmed.float())
    
    avg_loss = loss / batch_lengths.sum().float()
    return avg_loss

def train(model,loss_func,optimizer,data,epochs):
  
  for epoch in range(epochs):
    total_loss = 0
    for batch_inputs,batch_labels,batch_lengths in data:
      optimizer.zero_grad()
      forward = model.forward(batch_inputs)

      loss = loss_func(forward,batch_labels,batch_lengths)
      loss.backward()
      optimizer.step()

      total_loss+=loss.item()

    print(total_loss)



# train(model,loss_func,optimizer,data,20)


def evaluate(model, val_data):
    
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for batch_inputs, batch_labels, batch_lengths in val_data:
            prob = model(batch_inputs)  
            
            predictions = (prob > 0.5).long()
            
            max_output_len = prob.size(1)
            batch_labels_trimmed = batch_labels[:, :max_output_len]
            
            preds_flat = predictions.cpu().numpy().flatten()
            labels_flat = batch_labels_trimmed.cpu().numpy().flatten()
            inputs_flat = batch_inputs[:, :max_output_len].cpu().numpy().flatten()
            
            pad_id = model.pad_id
            valid_indices = (inputs_flat != pad_id)
            
            all_predictions.extend(preds_flat[valid_indices])
            all_ground_truths.extend(labels_flat[valid_indices])
            
    all_ground_truths = np.array(all_ground_truths)
    all_predictions = np.array(all_predictions)
    
    precision = precision_score(all_ground_truths, all_predictions, zero_division=0)
    recall = recall_score(all_ground_truths, all_predictions, zero_division=0)
    f1 = f1_score(all_ground_truths, all_predictions, zero_division=0)
    
    total_entities = np.sum(all_ground_truths == 1)
    detected_entities = np.sum((all_predictions == 1) & (all_ground_truths == 1))
    false_positives = np.sum((all_predictions == 1) & (all_ground_truths == 0))
    
    print("\n================ VALIDATION METRICS ================")
    print(f"Total True Entity Tokens in Val Set : {total_entities}")
    print(f"Correctly Detected Entity Tokens    : {detected_entities}")
    print(f"False Positives (Spurious Alarms)   : {false_positives}")
    print("----------------------------------------------------")
    print(f"Precision : {precision:.4f}  (Out of all predicted entities, how many were real?)")
    print(f"Recall    : {recall:.4f}  (Out of all real entities, how many did we catch?)")
    print(f"F1-Score  : {f1:.4f}  (Harmonic mean of Precision and Recall)")
    print("====================================================")
    
    model.train()
    return f1


val_loader = DataLoader(
    list(zip(val_sentences, val_labels)),
    batch_size=2,
    shuffle=False,  
    collate_fn=partial(custom_collate_fn, window_size=2, word2id=word2id)
)

# evaluate(model, val_loader)

