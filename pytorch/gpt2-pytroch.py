import numpy as np
import torch
import torch.nn as nn
import pandas as pd

class GPT2Model(nn.Module):
    def __init__(self, embedDim, vocabSize, windowSize):
        super().__init__()
        self.embedDim = embedDim
        self.customEmbed = CustomEmbed(embedDim, vocabSize, windowSize)
        nn.MultiheadAttention()


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