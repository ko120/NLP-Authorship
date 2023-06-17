
import torch
from transformers import AutoTokenizer, T5EncoderModel, AutoConfig
from torch.utils.data import DataLoader, Dataset


class MultiClassClassificationDataset(Dataset):
    def __init__(self, codes, labels, max_length= 512):
        self.codes = codes
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(code, return_tensors='pt', truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        sample = {'input_ids' : input_ids, 'labels' : label, 'attention_mask' : attention_mask}
        return sample
        
def collate_fn(batch):
    batch_data = {'input_ids': [], 'labels': [], 'attention_mask': []}
    
    max_input_len = max([b['input_ids'].size(1)for b in batch]) # Finding max length for padding 

    for b in batch:
        batch_data['input_ids'].append(torch.cat([b['input_ids'], torch.zeros(1, max_input_len - b['input_ids'].size(1)).long()], dim=1))
        batch_data['labels'].append(b['labels'])
        batch_data['attention_mask'].append(torch.cat([b['attention_mask'], torch.zeros(1, max_input_len - b['attention_mask'].size(1))], dim=1))
        
    batch_data['input_ids'] = torch.cat(batch_data['input_ids'], dim=0)
    batch_data['attention_mask'] = torch.cat(batch_data['attention_mask'], dim=0)
    return batch_data

