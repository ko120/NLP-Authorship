import torch
import transformers
import datasets
from transformers import AutoTokenizer, T5EncoderModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import pdb
import numpy as np
import time
import wandb
from datasets import load_dataset # This is from Hugging face
from dataset import MultiClassClassificationDataset, collate_fn # This my own class
from T5Classifer import T5Classifier
import os

def validation_step(model, test_datalaoder, device, criterion, epoch, print_interval, wandb_on, best_score):
  validation_loss = []
  validation_acc = []
  model.eval()
  accuracy = datasets.load_metric('accuracy')

  with torch.no_grad():
    for data in test_datalaoder:
          input_ids = data['input_ids'].to(device)
          attention_masks = data['attention_mask'].to(device)
          labels = torch.LongTensor(data['labels']).to(device)
          outputs = model(input_ids=input_ids, attention_mask=attention_masks)
          loss = criterion(outputs, labels)
          validation_loss.append(loss.mean().item())
          ac = accuracy.compute(predictions = outputs.argmax(-1), references = labels)
          validation_acc.append(ac['accuracy'])

    running_val_loss = round(sum(validation_loss)/len(validation_loss),4)
    running_val_acc = round(sum(validation_acc)/len(validation_acc),2)

    
    if wandb_on:
      wandb.log({'epoch': epoch, 'validation loss': running_val_loss, 'validation acc': running_val_acc})

    if epoch % print_interval == 0:
      print('validation loss {} validation_acc {}'.format(running_val_loss, running_val_acc *100 ))
      # Saving Validation Model
      if wandb_on:
        if running_val_acc > best_score:
          model.save('validation_model' + str(epoch))
          wandb.save('validation_model' + str(epoch))
          torch.save(model.state_dict(), 'validation_model_layer' + str(epoch))
          wandb.save('validation_model_layer' + str(epoch))

    if running_val_acc > best_score:
      best_score = running_val_acc
    
    model.train()
    return best_score


def train():
  # Setting GPU environment 
  os.environ["CUDA_VISIBLE_DEVICES"] = "2" # make use of only gpu 2 
  os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # prevent using gpu 1

  # For debugging purpose 
  if debug:
    wandb_on = False
  else:
    wandb_on = True

  # Initializing with Wandb
  if wandb_on:
    wandb.init(config = {'batch': 32,'num_authors':10}, group = '10 authors_large',settings=wandb.Settings(start_method="thread"))
    config = wandb.config
    batch_size = config.batch
    learning_rate = config.learning_rate
    num_authors = config.num_authors
  else:
    batch_size = 16
    learning_rate = 5e-5
    num_authors = 10


  # Setting the seed for reproduction
  torch.manual_seed(543)

  # Read data from json file
  jsonl_files = {
      'train': '/home/ko120/Authorship/NLP-Authorship/data/10-raw-train.jsonl',
      'test': '/home/ko120/Authorship/NLP-Authorship/data/10-raw-test.jsonl'
  }
  data = load_dataset('json', data_files = jsonl_files )

  # Split data
  #train_code, test_code, train_labels, test_labels = train_test_split(data_code, data_author, test_size=0.2, random_state=52)

  train_dataset = MultiClassClassificationDataset(data['train']['inputs'], data['train']['label'])
  test_dataset = MultiClassClassificationDataset(data['test']['inputs'], data['test']['label'])
  
  # Load data
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory= True, num_workers=0)
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory= True, num_workers=0)


  # Build model
  model_config = AutoConfig.from_pretrained('Salesforce/codet5-small')
  T5Encoder = T5EncoderModel.from_pretrained('Salesforce/codet5-small')
  model = T5Classifier(T5Encoder, model_config.hidden_size, num_classes = num_authors, dropout = 0.2)

  # Using GPU 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  num_epochs = 300
  print_interval = 50

  
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  criterion = nn.NLLLoss()
  scheduler = get_cosine_schedule_with_warmup(
          optimizer=optimizer, num_warmup_steps=0, num_training_steps=int(num_epochs * len(train_dataloader))
      )
  accuracy = datasets.load_metric('accuracy')
  best_score = 0.0

  if wandb_on:
    wandb.watch(model)

  for epoch in range(num_epochs + 1):
      total_loss = 0.0
      total_correct = 0
      oom = 0
      training_loss = []
      training_acc = []
      start_time = time.time()
      model.train()
      for data in train_dataloader:
          input_ids = data['input_ids'].to(device)
          attention_masks = data['attention_mask'].to(device)
          labels = torch.LongTensor(data['labels']).to(device)
         
          try:
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_masks)
            loss = criterion(outputs, labels)
            
            loss.mean().backward()
            nn.utils.clip_grad_value_(model.parameters(),0.3)
            optimizer.step()
            scheduler.step()

            training_loss.append(loss.mean().item())
           
            ac = accuracy.compute(predictions = outputs.argmax(-1), references = labels)
            training_acc.append(ac['accuracy'])
          except Exception as e: # When memory issue occurs, empty the model
            if 'out of memory' in str(e):
              oom +=1
            print('Memory issue occured')
            model.zero_grad()
            optimizer.zero_grad()
            del input_ids, attention_masks, labels
            break
     

      epoch_ac =  sum(training_acc) / len(training_acc)
      epoch_loss = sum(training_loss) / len(training_loss)
      if wandb_on:
        wandb.log({'epoch': epoch, 'epoch_ac': epoch_ac, 'epoch_loss': epoch_loss})
      if epoch % print_interval == 0:
        print("Epoch {}, Loss={}, Accuracy={}  time={} num of out of memory issue {}".format(epoch, round(epoch_loss,4),round(epoch_ac*100,2), int(time.time() - start_time), oom))
        oom = 0
      
      # Validation during training  
      best_score = validation_step(model, test_dataloader, device, criterion, epoch, print_interval,wandb_on, best_score)

if __name__ == '__main__':
  debug = True
  if not debug:
    sweep_config = dict()
    sweep_config['method'] = 'grid'
    sweep_config['metric'] = {'name': 'epoch acc', 'goal': 'maximize'}
    sweep_config['parameters'] = {'learning_rate': {'values' : [2e-5,3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5]}, 'optimizer': {'values' : ['AdamW']}}

    sweep_id = wandb.sweep(sweep_config, project = 'Authorship_NLP',)
    wandb.agent(sweep_id, train)
  else:
    train()
  