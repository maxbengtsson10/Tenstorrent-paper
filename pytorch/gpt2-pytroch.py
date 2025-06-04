import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time
from tqdm import tqdm


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

    
    def forward(self, text, padMask, i):
        while i < self.windowSize:
            x = self.customEmbed(text)
            for layer in self.transformerLayers:
                x = layer(x,padMask, self.autoRegMask)
            nextTok = x[:,-1,:]
            nextTok = self.outputLayer(nextTok)
            nextTok = self.softmax(nextTok)
            nextTokID = torch.argmax(nextTok,dim=-1)
            
            text[:,i] = nextTokID
            padMask[:,i] = False
            i += 1


if __name__ == "__main__":

    device = "mps"
    embedDim = 768
    heads = 12
    vocab = 50000
    windowSize = 512 
    layers = 12

    runs = 5


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


    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    timeStart = time.perf_counter()

    with torch.no_grad():
        for _ in tqdm(range(runs)):
            inputWordUse = inputWord.clone()
            inputMaskUse = inputMask.clone()
            model(inputWordUse,inputMaskUse,1)

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    timeEnd = time.perf_counter()

    totalTime = timeEnd - timeStart  # in seconds, often sub‐microsecond precision
    print(f"Total Time: {totalTime:.9f} sec")
    print(f"Per Run: {totalTime/runs:.9f} sec")
    print(f"Tokens/Sec: {((windowSize-1)*runs)/totalTime:.2f}")



