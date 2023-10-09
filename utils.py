import torch
import nltk
import torch.nn as nn


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