import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time
from tqdm import tqdm

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met


class CustomEmbed(nn.Module):
    def __init__(self, embedDim, vocabSize, windowSize, device):
        super().__init__()
        self.embed = nn.Embedding(vocabSize,embedDim)
        self.embedDim = embedDim
        self.pe = torch.zeros((windowSize,embedDim))
        for word in range(windowSize):
            for i in range(embedDim // 2):
                self.pe[word,2*i] = np.sin(word/(np.power(10000,((2*i)/embedDim))))
                self.pe[word,2*i+1] = np.cos(word/(np.power(10000,((2*i)/embedDim))))
        self.pe = self.pe.to(device)

    
    def forward(self, vector):
        embed = self.embed(vector)
        positionalEmbed = embed + self.pe.expand(embed.shape[0],-1,-1)
        return positionalEmbed


class CustomTransformer(nn.Module):
    def __init__(self,embedDim,numHeads):
        super().__init__()

        self.Q = nn.Linear(embedDim,embedDim)
        self.K = nn.Linear(embedDim,embedDim)
        self.V = nn.Linear(embedDim,embedDim)

        self.mha = nn.MultiheadAttention(embed_dim=embedDim,num_heads=numHeads,batch_first=True)
        
        self.ffnn = nn.Sequential(
            nn.LayerNorm(embedDim),
            nn.Linear(embedDim,4*embedDim),
            nn.GELU(),
            nn.Linear(embedDim*4,embedDim),
            nn.GELU()
        )

    def forward(self,embed, padMask, attn):

        q = self.Q(embed)
        k = self.K(embed)
        v = self.V(embed)
        x, _ = self.mha(q,k,v,attn_mask=attn,key_padding_mask=padMask,is_causal=True)
        out = self.ffnn(x)
        return out





class GPT2Model(nn.Module):
    def __init__(self, embedDim, numHeads, vocabSize, windowSize, layers, device):
        super().__init__()
        self.embedDim = embedDim
        self.layers = layers
        self.windowSize = windowSize
        self.customEmbed = CustomEmbed(embedDim, vocabSize, windowSize, device)
        
        self.transformerLayers = nn.ModuleList([CustomTransformer(embedDim,numHeads) for _ in range(layers)])

        self.autoRegMask = nn.Transformer.generate_square_subsequent_mask(self.windowSize,dtype=torch.bool)
        self.autoRegMask = self.autoRegMask.to(device)

        self.outputLayer = nn.Linear(embedDim,vocabSize)
        self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, text, padMask):
        x = self.customEmbed(text)
        for layer in self.transformerLayers:
            x = layer(x,padMask, self.autoRegMask)
        nextTok = x[:,-1,:]
        nextTok = self.outputLayer(nextTok)
        nextTok = self.softmax(nextTok)
        nextTokID = torch.argmax(nextTok,dim=-1)

        return nextTokID


if __name__ == "__main__":

    device = xm.xla_device()
    print(f"Using device: {device}")
    
    embedDim = 768
    heads = 12
    vocab = 50000
    windowSize = 512 
    layers = 12



    model = GPT2Model(embedDim,heads,vocab,windowSize,layers,device)
    model.eval()
    model.to(device)

    inputWord = torch.zeros(windowSize,dtype=torch.long)
    inputWord[0] = np.random.randint(0,50000)
    inputWord = inputWord.unsqueeze(0)
    inputWord = inputWord.to(device)

    mask = [True for _ in range(windowSize)]
    mask[0] = False
    inputMask = torch.tensor(mask)
    inputMask = inputMask.unsqueeze(0)
    inputMask = inputMask.to(device)


    print("Warming up TPU...")
    with torch.no_grad():
        i = 0
        for i in range(5):
            nextToken = model(inputWord, inputMask)
            inputWord[:,i] = nextToken
            inputMask[:,i] = False
    
    xm.mark_step()
    print("TPU warmup complete")

    inputMask[:,0] = False
    inputMask[:,1:] = True

    timeStart = time.perf_counter()

    for i in tqdm(range(1,windowSize)):
        nextTok = model(inputWord,inputMask)
        inputWord[:,i] = nextTok
        inputMask[:,i] = False

        if i % 10 == 0:
            xm.mark_step()

    xm.mark_step()
    timeEnd = time.perf_counter()

    

    totalTime = timeEnd - timeStart  # in seconds, often sub‐microsecond precision
    print(f"Total Time: {totalTime:.9f} sec")
    print(f"Tokens/Sec: {((windowSize-1))/totalTime:.2f}")



