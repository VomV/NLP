from torch import nn, bmm, F
from math import sqrt

#mask is a lower triangular matrix with 1s and 0s

def scaled_dotProdAttn_masked(query, key, value, mask=None):

    '''
        Returns contextualized embeddings
    '''

    dim_k = query.size(-1)
    scores = bmm(query, key.transpose(1,2))//sqrt(dim_k)

    if mask is not None:
        scores = scores.masked_fill(mask==0, float("-inf"))
    
    weights = F.softmax(scores, dim=-1)
    return weights.bmm(value)

