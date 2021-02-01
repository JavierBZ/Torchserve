
import torch
import torch.nn as nn

from transformers import BertModel

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes,PRE_TRAINED_MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'):
    super(SentimentClassifier, self).__init__()


    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    H = 50
    # Instantiate an one-layer feed-forward classifier
    self.classifier = nn.Sequential(
        nn.Linear(self.bert.config.hidden_size,H),
        nn.ReLU(),
        #nn.Dropout(0.5),
        nn.Linear(H, n_classes)
    )

    #self.out = nn.Linear(, n_classes)

  def forward(self, input_ids, attention_mask):
    _, output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    ).to_tuple()
    if self.training:
      output = self.drop(output)
    return self.classifier(output),output
