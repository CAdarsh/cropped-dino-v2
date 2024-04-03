import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class SelfAttention(nn.Module):
    def qkv_head(self, indim, hidden_dim, nheads):
        return nn.Sequential(nn.Linear(indim, hidden_dim, bias=False),
            Rearrange('bsize indim (nheads hidden_dim) -> bsize nheads indim hidden_dim', nheads=nheads))
        
    def __init__(self, indim, att_dim, nheads, dropout):
        super(SelfAttention, self).__init__()
        hidden_dim = att_dim * nheads
        self.scale = hidden_dim ** -0.5

        self.key_head = self.qkv_head(indim, hidden_dim, nheads)
        self.query_head = self.qkv_head(indim, hidden_dim, nheads)
        self.value_head = self.qkv_head(indim, hidden_dim, nheads)

        self.attention_scores = nn.Softmax(dim=-1)
        self.droput = nn.Dropout(dropout)

        self.out_layer = nn.Sequential(Rearrange('bsize nheads indim hidden_dim -> bsize indim (nheads hidden_dim)'),
                         nn.Linear(hidden_dim, indim),
                         nn.Dropout(dropout))

    def forward(self, x):
        query = self.query_head(x)
        key = self.key_head(x)
        value = self.value_head(x)

        dotp = torch.matmul(query, key.transpose(-1, 2)) * self.scale

        scores = self.attention_scores(dotp)

        scores = self.droput(scores)

        weighted_scores = torch.matmul(scores, value)

        out = self.out_layer(weighted_scores)

        return out

class Encoder(nn.Module):
    def __init__(self, nheads, nlayers, embed_dim, head_dim, mlp_hidden_dim, dropout):
        super(Encoder, self).__init__()
        self.nheads = nheads
        self.nlayers = nlayers
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout = dropout

        self.att_layers, self.forward_layers = self.get_layers()

    def get_layers(self):
        att_modules = nn.ModuleList()
        forward_modules = nn.ModuleList()

        for i in range(self.nlayers):
            att_modules.append(nn.Sequential(nn.LayerNorm(self.embed_dim), SelfAttention(self.embed_dim, self.head_dim, self.nheads, self.dropout)))
            forward_modules.append(nn.Sequential(nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, self.mlp_hidden_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(self.mlp_hidden_dim, self.embed_dim), nn.Dropout(self.dropout)))

        return att_modules, forward_modules

    def forward(self, x):
        for (att_layer, forward_layer) in zip(self.att_layers, self.forward_layers):
            x = x + att_layer(x)
            x = x + forward_layer(x)
        
        return x
    
class VanillaVit(nn.Module):
    def __init__(self, cfg):
        super(VanillaVit, self).__init__()

        input_size = cfg['input_size']
        self.patch_size = cfg['patch_size']
        self.embed_dim = cfg['embed_dim']
        att_layers = cfg['att_layers']
        nheads = cfg['nheads']
        head_dim = cfg['head_dim']
        mlp_hidden_dim = cfg['mlp_hidden_dim']
        dropout = cfg['dropout']
        nclasses = cfg['nclasses']
        
        self.num_patches = (input_size[0] // self.patch_size) * (input_size[1] // self.patch_size) + 1

        self.patch_embedding = nn.Sequential(Rearrange('b c (h px) (w py) -> b (h w) (px py c)', px= self.patch_size, py=self.patch_size), nn.Linear(self.patch_size * self.patch_size * 3, self.embed_dim))

        self.dropout = nn.Dropout(dropout)

        self.class_token = nn.Parameter(torch.randn(1,1, self.embed_dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim))

        self.transformer = Encoder(nheads=nheads, nlayers = att_layers, embed_dim=self.embed_dim, head_dim = head_dim, mlp_hidden_dim=mlp_hidden_dim, dropout=dropout)

        self.pred_head = nn.Sequential(nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, nclasses))

    def forward(self, x):
        npatches = (x.size(2) // self.patch_size) * (x.size(3) // self.patch_size) + 1
        
        embedding = self.patch_embedding(x)

        x = torch.cat((self.class_token.repeat(x.size(0), 1, 1), embedding), dim=1)

        if npatches == self.num_patches:
            x += self.pos_embedding
        else:
            interpolated = nn.functional.interpolate(self.pos_embedding[None], (npatches, self.embed_dim), mode='bilinear')
            x += interpolated[0]
        
        x = self.dropout(x)
        x = self.transformer(x)
        x = x[:, 0, :]
        
        return self.pred_head(x)
    