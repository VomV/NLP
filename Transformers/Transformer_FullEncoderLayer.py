from torch import nn, bmm
import torch

from math import sqrt
from transformers import AutoConfig, AutoTokenizer


model_name = 'bert-base-uncased'
config = AutoConfig.from_pretrained(model_name)


#Scaled dot prod attn
def scaled_dot_prod_attn(query, key, value):

    dim_k = query.size(-1)
    scores = bmm(query, key.transpose(1,2))/sqrt(dim_k)
    weights = nn.functional.softmax(scores, dim=-1)
    attn_outputs = bmm(weights, value)
    return attn_outputs


#Feed Forward Layer
class FeedForward(nn.Module):

    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


#Attention Head
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim) -> None:
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_prod_attn(self.q(hidden_state),
                                                    self.k(hidden_state),
                                                    self.v(hidden_state))
        return attn_outputs


#Multi headed attention
class MultiAttentionHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim//num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x
        

#Encoder layer
#Applying layer normalization before each layer
class TransformerEncoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiAttentionHead(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        #apply layer norm then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)

        #apply attn with skip connection
        x = x + self.attention(hidden_state)

        #apply layer norm again
        x_1 = self.layer_norm_2(x)

        #apply feed forward layer with skip connection
        x = x + self.feed_forward(x_1)

        return x


#Test with inputs
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = 'Its better to take action than wonder'
inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)

config = AutoConfig.from_pretrained(model_name)
