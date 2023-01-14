from torch import nn
import Transformer_FullEncoderLayer as el
from transformers import AutoConfig, AutoTokenizer

class TransformerForSequenceClassification(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = el.TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.encoder(x)[:, 0, :] #hidden state of cls token
        x = self.dropout(x)
        x = self.classifier(x)
        return x

model_name = 'bert-base-uncased'
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = 'Time for some classification'
inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)

config.num_labels = 3
encoder_classifier = TransformerForSequenceClassification(config)
print(encoder_classifier(inputs.input_ids).size())


