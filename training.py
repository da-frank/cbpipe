import sys, os
import json
from stopwords import worte

# Importiere PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Natural Language Toolkit
import nltk
from nltk.stem.lancaster import LancasterStemmer
nltk.download('punkt')

# Initialisiere Stemmer
STEMMER = LancasterStemmer()

# Herzstück für das Textverständnis
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

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================ Datenverarbeitung und Vorbereitung
# HIER DATEI REINWERFEN
with open("intents.json") as file:
    data = json.load(file)

words = []  # Wörter, die der Chatbot erkennen können soll
labels = [] # zugehörige Labels (siehe Output unten)
docs_x = [] # Trainingsgedöhns
docs_y = []

# Durchlaufe die Intents
for intent in data["intents"]:
    # Speichere Pattern-Token (gekürzte Wörter) mit zugehörigen Labeln
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
    
    if intent["tag"] not in labels:
        labels.append(intent["tag"])


words = [w for w in words if not w in worte] # Schmeiße Stopwords raus (sowas wie "als" oder "habe"), die irrelevant für die Klassifizierung sind
words = [STEMMER.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]


# Generiere training und output für Training des Chatbots
for x, doc in enumerate(docs_x):
    bag = []
    wrds = [STEMMER.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)

training = torch.tensor(training).float().to(device)
output = torch.tensor(output).float().to(device)


# ====================================== "Gehirn"


# Hier sollen die Studierenden rumspielen.
# Aktuell: Ein MLP mit einem Layer und ohne Aktivierungsfunktion
class Classifier(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layer1 = nn.Linear(dim_in, dim_out)
    
    def forward(self, x):
        out = self.layer1(x)
        return out

dim_in = len(training[0])
dim_out = len(output[0])
model = Classifier(dim_in, dim_out).to(device)

# ====================================== Training "Gehirn"

# Trainiere das Chatbot-Gehirn
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_func = F.cross_entropy

n_epochs = 5000

# Da der Datensatz nur 525 Einträge enthält, brauchen wir keine Batches und können komplett trainieren
for epoch in range(n_epochs):
    optimizer.zero_grad()
    out = model(training)
    loss = loss_func(out, output)
    loss.backward()
    optimizer.step()

# EXPORTIEREN DES MODELS
torch.save(model.state_dict, "demofile.pt")