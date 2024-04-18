# utils.py
# - Enthält Hilfsfunktionen für das Training und die Evaluation des Modells

import torch
import json
import nltk
import torch.nn as nn
from random import shuffle


class Classifier(nn.Module):
    # MLP für Chatbot Melinda
    # Hier: Standard-MLP-Architektur
    # Im ChaBoDoc-Projekt: 
    #   Immer drei Layer:
    #       - Eingabelayer mit Neuronenzahl = Anzahl bekannter Wörter
    #       - Hidden-Layer mit halber Neuronenzahl vom Eingabelayer (abgerundet)
    #       - Ausgabelayer mit Anzahl der Klassen (bei ChaBoDoc sind es immer 3: good, bad, neutral)
    def __init__(self, dims=[]):
        super().__init__()
        layerlist = []
        for i in range(len(dims) - 1):
            layerlist.append(nn.Linear(dims[i], dims[i + 1]))
            layerlist.append(nn.ReLU())
        self.layers = nn.Sequential(*(layerlist[:-1]))

    def forward(self, x):
        out = self.layers(x)
        return out

# Alt, wird vermutlich nicht mehr direkt gebraucht
def bagofwords(s, words):
    # Input: Satz s (User-Input), Liste bekannter Wörter words
    # Output: Vektor mit Nullen und Einsen
    bag = [0 for _ in range((len(words)))]
    s_words = nltk.word_tokenize(s) # Splitte Satz auf in einzelne Wörter und Satzzeichen
    s_words = [STEMMER.stem(word.lower()) for word in s_words] # "Kürze" Wörter gemäß Lancaster-Stemmer
    print(s_words)

    # Wenn Wort in Wortliste enthalten, setze 1, sonst 0
    for se in s_words:
        for i, w in enumerate(words):
            if w==se:
                bag[i] = 1
        
    return torch.tensor(bag).float() 

class Dataset(torch.utils.data.Dataset):
    def __init__(self, model, tokenizer, name="melinda", timestamp="", train=True, split=0.8):
        super().__init__()
        with open(f"inputs/{name}_{timestamp}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        labeldict = {"neutral words": 2, "bad words": 1, "good words": 0}
        self.data = []
        self.label = []
        for d in data["intents"]:
            temp = d["patterns"]
            shuffle(temp)
            if train:
                self.data.extend(temp[:int(split*len(temp))])
                self.label.append(torch.ones(len(d["patterns"]))[:int(split*len(temp))] * labeldict[d["tag"]])
            else: 
                self.data.extend(temp[int(split*len(temp)):])
                self.label.append(torch.ones(len(d["patterns"]))[int(split*len(temp)):] * labeldict[d["tag"]])
        self.label = torch.cat(self.label)
        print(f"Good: {sum(self.label == 1)}, Bad: {sum(self.label == 0)}, Neutral: {sum(self.label == 2)}")

        with torch.no_grad():
            ttt = tokenizer(self.data, return_tensors="pt", padding=True, truncation=True)
            self.data = model(ttt.input_ids, attention_mask=ttt.attention_mask).logits
        
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.label)
    

class Dataset_Mogelbot(torch.utils.data.Dataset):
    def __init__(self, model, tokenizer, name="mogelbot"):
        super().__init__()
        with open("data/{}.json".format(name), "r", encoding="utf-8") as f:
            data = json.load(f)
        
        labeldict = {"neutral words": 2, "bad words": 1, "good words": 0}
        self.data = []
        self.label = []
        for d in data["intents"]:
            self.data.extend(d["patterns"])
            self.label.append(torch.ones(len(d["patterns"])) * labeldict[d["tag"]])
        self.label = torch.cat(self.label)
        print(f"Good: {sum(self.label == 1)}, Bad: {sum(self.label == 0)}, Neutral: {sum(self.label == 2)}")

        with torch.no_grad():
            ttt = tokenizer(self.data, return_tensors="pt", padding=True, truncation=True, max_length=max([len(d) for d in self.label]))
            self.data = model(ttt.input_ids, attention_mask=ttt.attention_mask).logits
        
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.label)
    
if __name__ == '__main__':
    ds = Dataset()
    print(len(ds))
    print(ds[100])
