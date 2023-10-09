import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk
STEMMER = nltk.stem.lancaster.LancasterStemmer()
nltk.download('punkt')

import os
import json

PATHDATA = "./"
PATHSTOP = "./" # PATHDATA+"stopwords/"
PATHWORDS = PATHDATA+"words/"
PATHNET = PATHDATA+"outputs/networks/state_dicts/"
print(os.getcwd())

class Classifier(nn.Module):
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


def bagofwords(s, words, stopwords):
    # Input: Satz s (User-Input), Liste bekannter Wörter words
    # Output: Vektor mit Nullen und Einsen
    bag = [0 for _ in range((len(words)))]
    #print("baglen", len(bag))
    s_words_tokenize = nltk.word_tokenize(
        s
    )  # Splitte Satz auf in einzelne Wörter und Satzzeichen
    s_words = [
        STEMMER.stem(word.lower())
        for word in s_words_tokenize
        if word.lower() not in stopwords
    ]  # "Kürze" Wörter gemäß Lancaster-Stemmer
    # Wenn Wort in Wortliste enthalten, setze 1, sonst 0
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return (
        s_words_tokenize,
        s_words,
        torch.tensor(bag).float(),
    )

def Netpicker(name):
    if name=="Frankensteinmonster":
        net = Frankensteinmonster_Net()
    elif name=="Mogelnet":
        net = Mogelnet()
    else:
        net = Gruppe(name)
    return net

def entropy(liste):
    liste += 1e-10
    out = -torch.sum(liste*liste.log())
    return out
    
class Frankensteinmonster_Net():
    def __init__(self, gruppen=["Melinda", "Salzwerk", "Gruppe", "MarzInator", "LuSo", "Supernet"]):
        self.name = "Frankensteinmonster"
        self.netlist = []
        for gruppe in gruppen:
            self.netlist.append(Gruppe(gruppe))
        
    def predict(self, x):
        predlist = torch.zeros([len(self.netlist),3])
        entropylist = []
        for i, gruppe in enumerate(self.netlist):
            pred = gruppe.predict(x)
            predlist[i] = pred
            entropylist.append(entropy(pred))
        
        entropylist = entropy(torch.tensor([1/3,1/3,1/3]))-torch.tensor(entropylist)
        entropylist *= 1/entropylist.sum()
        out = predlist.T*(entropylist)
        return out.sum(dim=-1).squeeze()

class Mogelnet():
    def __init__(self):
        self.name = "Mogelnet"
        path = PATHDATA+"/Mogelnet/"
        with open(path+"data.json", "r", encoding="utf-8") as file:
            self.data = json.load(file)
        with open(PATHSTOP+"stopwords.txt", "r", encoding="utf-8") as file:
            self.stopwords = [w.replace("\n","") for w in file.readlines()]
    
    def predict(self, s):
        out = torch.zeros(3)
        s_words_tokenize = nltk.word_tokenize(
            s
        )  # Splitte Satz auf in einzelne Wörter und Satzzeichen
        nicht = False
        for s in s_words_tokenize:
            if s.lower()=="nicht":
                nicht=True
        s_words = [
            STEMMER.stem(word.lower())
            for word in s_words_tokenize
            if word.lower() not in self.stopwords
        ]
        print(s_words)
        for i, s in enumerate(s_words):
            if s in self.data["bad"]:
                out[0] += i+1
                out[1] -= i+1
            if s in self.data["good"]:
                out[0] -= i+1
                out[1] += i+1
            if s in self.data["neutral"]:
                out[2] += 1
            print(s, out)
        print(out)
        if nicht:
            out *= torch.tensor([-1, -1, 1])
        print(out)
        return F.softmax(out)

class Gruppe():
    def __init__(self, name, suffix=""):
        self.name = name
        self.suffix = suffix

        with open(
            PATHWORDS + self.name + self.suffix + "_words.txt", "r", encoding="utf-8"
        ) as file:
            self.words = [w.replace("\n", "").strip() for w in file.readlines() if w != "" or w != "\n"]
        
        self.words = [STEMMER.stem(w) for w in self.words]
        
        with open(PATHSTOP + "stopwords.txt", "r", encoding="utf-8") as file:
            self.stopwords = [w.replace("\n", "").strip() for w in file.readlines() if w != "" or w != "\n"]

        """with open(
            PATHSTOP + self.name + self.suffix + "_stop.txt", "r", encoding="utf-8"
        ) as file:
            self.stopwords.extend([w.replace("\n", "").strip() for w in file.readlines() if w != "" or w != "\n"])
"""
        #print(len(self.words))
        """for w in self.words:
            if w.lower() in self.stopwords:
                print(w)
        self.words = [w.lower() for w in self.words if not w.lower() in self.stopwords]"""
        #print(len(self.words))
        #self.network = torch.load(PATHNET + self.name + self.suffix + "_model.pt")
        self.network = Classifier(dims=[len(self.words), int(len(self.words) / 2), 3])
        self.network.load_state_dict(
            torch.load(PATHNET + self.name + self.suffix + ".pt", map_location=torch.device("cpu"))
        )

    def predict(self, input):
        s_words_tokenize, s_words, bag = bagofwords(input.lower(), self.words, self.stopwords)
        """print(bag.sum())
        print(s_words_tokenize)
        print(s_words)
        #print(self.words)

        print([w for i, w in enumerate(self.words) if bag[i]>0])"""
        result = F.softmax(self.network(bag), dim=-1)
        return result