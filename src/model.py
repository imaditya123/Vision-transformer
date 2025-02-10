import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from tqdm import tqdm

from utils import initialize_weights


class Attention(nn.Module):
  def __init__(self,embed_dim,head_dim,dropout_rate):
    super(Attention,self).__init__()

    self.query=nn.Linear(embed_dim,head_dim)
    self.key=nn.Linear(embed_dim,head_dim)
    self.value=nn.Linear(embed_dim,head_dim)
    self.dropout=nn.Dropout(dropout_rate)

    initialize_weights(self.query)
    initialize_weights(self.key)
    initialize_weights(self.value)


  def forward(self,query,key,value,mask=None):
    d_k=query.size(-1)

    q=self.query(query)
    k=self.key(key)
    v=self.value(value)

    scores=q @ k.transpose(1,2) /math.sqrt(d_k)
    if mask is not None:
      # mask = mask.unsqueeze(1)
      scores = scores.masked_fill(mask == 0, float('-inf'))

    weights=F.softmax(scores,dim=-1)
    weights=self.dropout(weights)
    out=weights @ v
    return out
  
class MultiheadAttention(nn.Module):
  def __init__(self,embed_dim,head_size,dropout_rate):
    super(MultiheadAttention,self).__init__()

    # Validate that embed_dim is divisible by head_size
    assert embed_dim % head_size == 0, "embed_dim must be divisible by head_size"

    self.head_dim=embed_dim//head_size
    self.embed_dim=embed_dim
    self.head_size=head_size

    self.attn_heads=nn.ModuleList([Attention(embed_dim,self.head_dim,dropout_rate) for _ in range(head_size)])
    self.out_layer=nn.Linear(self.head_dim*head_size,embed_dim)
    self.dropout=nn.Dropout(dropout_rate)

    initialize_weights(self.out_layer)


  def forward(self,query,key,value,mask=None):

    out=torch.cat([h(query,key,value,mask) for h in self.attn_heads],dim=-1)
    out=self.dropout(self.out_layer(out))
    return out

class ViTMLP(nn.Module):
  def __init__(self,hidden_dim,filter_size,dropout=0.5):
    super(ViTMLP,self).__init__()
    self.linear1=nn.Linear(hidden_dim,filter_size)
    self.gelu=nn.GELU()
    self.dropout1=nn.Dropout(dropout)
    self.linear2=nn.Linear(filter_size,hidden_dim)
    self.dropout2=nn.Dropout(dropout)

    initialize_weights(self.linear1)
    initialize_weights(self.linear2)

  def forward(self,x):
    x=self.linear1(x)
    x=self.gelu(x)
    x=self.dropout1(x)
    x=self.linear2(x)
    x=self.dropout2(x)
    return x
  
class PatchEmbedding(nn.Module):
  def __init__(self,img_size=96,patch_size=16,hidden_dim=512):
    super(PatchEmbedding,self).__init__()

    self.num_patches = (img_size // patch_size) ** 2
    self.conv=nn.LazyConv2d(hidden_dim,kernel_size=patch_size,stride=patch_size)

  def forward(self,x):
    return self.conv(x).flatten(2).transpose(1,2)
  
class VitBlock(nn.Module):
  def __init__(self,hidden_dim,norm_shape,filter_size,num_heads,dropout):
    super(VitBlock,self).__init__()
    self.attn_norm=nn.LayerNorm(norm_shape)
    self.attn=MultiheadAttention(embed_dim=hidden_dim,head_size=num_heads,dropout_rate=dropout)

    self.mlp_norm=nn.LayerNorm(norm_shape)
    self.mlp=ViTMLP(hidden_dim=hidden_dim,filter_size=filter_size,dropout=dropout)


  def forward(self,x,valid_lens=None):
    y=self.attn_norm(x)
    y=self.attn(y,y,y,valid_lens)
    x=x+y

    y=self.mlp_norm(x)
    y=self.mlp(y)
    x=x+y

    return x

class ViT(nn.Module):
  def __init__(self,img_size,patch_size,hidden_dim,filter_size,num_heads,n_layers,dropout_rate,lr=0.1,num_classes=10):
    super(ViT,self).__init__()
    self.patch_embedding=PatchEmbedding(img_size=img_size,patch_size=patch_size,hidden_dim=hidden_dim)

    self.cls_token=nn.Parameter(torch.zeros(1,1,hidden_dim))
    num_steps=self.patch_embedding.num_patches+1
    self.pos_embedding=nn.Parameter(torch.randn(1,num_steps,hidden_dim))

    self.dropout=nn.Dropout(dropout_rate)
    self.layers=nn.ModuleList([VitBlock(hidden_dim=hidden_dim,norm_shape=hidden_dim,filter_size=filter_size,num_heads=num_heads,dropout=dropout_rate) for _ in range(n_layers)])

    self.out=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,num_classes))


  def forward(self,x):
    x=self.patch_embedding(x)
    x=torch.cat((self.cls_token.expand(x.shape[0],-1,-1),x),1)
    x=self.dropout(x+self.pos_embedding)

    for layer in self.layers:
      x=layer(x)

    return self.out(x[:,0])