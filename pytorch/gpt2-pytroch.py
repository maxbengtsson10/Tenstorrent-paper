import numpy as np
import torch
import torch.nn as nn
import pandas as pd

def generateRandomData():
    pass

class CustomEmbed(nn.module):
    def __init__(self, embedDim, vocabSize, windowSize):
        super().__init__()
        self.embed = nn.Embedding(vocabSize,embedDim)
        self.embedDim = embedDim
        self.pe = torch.zeros((windowSize,embedDim))
        for word in range(windowSize):
            for i in range(embedDim/2-1):
                self.pe[word,2*i] = torch.sin(word/(torch.pow(10000,((2*i)/embedDim))))
                self.pe[word,2*i+1] = torch.cos(word/(torch.pow(10000,((2*i)/embedDim))))

    
    def forward(self, vector):
        embed = self.embed(vector)
        positionalEmbed = embed + self.pe.expand(embed.shape(0),-1,-1)
        return positionalEmbed



class GPT2Model(nn.Module):
    def __init__(self, embedDim, vocabSize, windowSize, layers):
        super().__init__()
        self.embedDim = embedDim
        self.layers = layers
        self.customEmbed = CustomEmbed(embedDim, vocabSize, windowSize)
        self.Q = nn.Linear(embedDim,embedDim)
        self.K = nn.Linear(embedDim,embedDim)
        self.V = nn.Linear(embedDim,embedDim)
        self.transformerBlock = nn.Sequential(
            nn.MultiheadAttention(embed_dim=embedDim,num_heads=12,attbatch_first=True,),
            nn.Linear(embedDim,4*embedDim),
            nn.GELU(),
            nn.Linear(embedDim*4,embedDim),
            nn.GELU()
        )
        self.outputLayer = nn.Linear(embedDim,vocabSize)
        self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, batch, attn, pad_mask):
        while sum(pad_mask > 0) > 0:
            embeded = self.customEmbed(batch)
            for _ in range(self.layers):
                self.transformerBlock(embeded, attn, )
            


