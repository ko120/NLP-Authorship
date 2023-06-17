import torch.nn as nn

class T5Classifier(nn.Module):
  def __init__(self, T5Encoder, hidden_size, num_classes, dropout= 0.2):
    super(T5Classifier, self).__init__()
    self.T5Encoder = T5Encoder
    self.dropout = dropout
    self.hidden_size = hidden_size
    self.num_classes = num_classes
    self.hidden_size = hidden_size 
    self.dropout = nn.Dropout(dropout)

    self.f1 = nn.Linear(self.hidden_size, self.num_classes)
    self.output_act = nn.LogSoftmax(dim=1)

  def forward(self, input_ids, attention_mask):

    out = self.T5Encoder(input_ids = input_ids, attention_mask = attention_mask)
    out = out[0]  # [B, L, H]
    out = out[:, 0, :]  # [B, H] taking first index since first index is CLS token 
    out = self.f1(out)
    out = self.dropout(out)
    out = self.output_act(out)
    return out

  def save(self, source):
    saved = self.T5Encoder.save_pretrained(str(source))
    return saved