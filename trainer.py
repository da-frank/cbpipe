# main.py
# Trainiert nacheinander alle Gruppenchatbots
import sys
from utils import Classifier  # ACHTUNG: utils.py wird hier benötigt!
import json
import os
import matplotlib.pyplot as plt
import nltk
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import trange
from torch import Tensor
from typing import List, Any

class Trainer:

    projectdir: str = "./"
    inputdir: str = f"{projectdir}inputs/"
    outputdir: str = f"{projectdir}outputs/"

    def __init__(self, stemmer="lancaster"):
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)

        nltk.download('punkt')

        # STEMMER (hier am besten Cistem statt Lancaster)
        if stemmer == "lancaster":
            self.STEMMER = nltk.stem.LancasterStemmer()
        elif stemmer == "cistem":
            self.STEMMER = nltk.stem.Cistem()
        else:
            print("Stemmer nicht gefunden")
            sys.exit()

    def train(self, gruppe: str) -> None:

        print(20*"-", "Training", gruppe, 20*"-")
        # Lade Trainingsdaten in den Arbeitsspeicher
        with open(f"{Trainer.inputdir}{gruppe}.json", encoding="utf-8") as file:
            intentsdata: Any = json.load(file)
            # print(intentsdata)
        # Lade Stopwords, falls sie existieren (die werden ignoriert)
        with open(f"{Trainer.projectdir}stopwords.txt", "r", encoding="utf-8") as file:
            # Allgemeine Stopwords
            stopwords: list[str] = [w.replace("\n", "").strip()
                        for w in file.readlines() if w != "" or w != "\n"]
        if f"{gruppe}_stop.txt" in os.listdir(f"{Trainer.inputdir}stopwords"):
            with open(f"{Trainer.inputdir}stopwords/{gruppe}_stop.txt", "r", encoding="utf-8") as file:
                # Gruppeneigene Stopwords
                stopwords.extend([w.replace("\n", "").strip()
                            for w in file.readlines() if w != "" or w != "\n"])
        stopwords = list(set([w.lower() for w in stopwords]))

        # Generiere Trainingsdaten
        words = []  # Liste bekannter Wörter in Tokenform
        labels = []  # zugehöriges Label (good, bad, neutral)
        # Liste aller Sätze im Datensatz (anders als words, da words nur einzelne Wörter enthält)
        docs_x = []
        docs_y = []  # Label zu docs_x
        for intent in intentsdata["intents"]:
            for pattern in intent["patterns"]:
                wrds: List[str] = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        print(len(words))
        # Stopwords aus words entfernen
        words: List[str] = [w.lower() for w in words if not w.lower() in stopwords]
        # Token in Wortstämme umwandeln
        words: List[str] = [self.STEMMER.stem(w.lower()) for w in words if w != "?"]
        words: List[str] = sorted(list(set(words)))  # list(set(...)) löscht doppelte Einträge
        print(len(words))

        with open(f"{Trainer.projectdir}words/{gruppe}_words.txt", "w", encoding="utf-8") as file:
            # Speichere bekannte Wörter ab
            file.writelines([w+"\n" for w in words])
        labels: List[str] = sorted(labels)
        print(labels)
        print(len(words))

        # Trainingsdaten vorbereiten
        training = []
        output = []
        out_empty: List[int] = [0 for _ in range(len(labels))]  # Lange Liste mit Nullen

        for x, doc in enumerate(docs_x):
            # bag: list[int] = []
            wrds: List[str] = [self.STEMMER.stem(w.lower()) for w in doc]

            # for w in words:
            #     if w in wrds:
            #         bag.append(1)
            #     else:
            #         bag.append(0)

            bag = [1 if w in wrds else 0 for w in words]

            output_row: List[int] = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1
            training.append(bag)
            output.append(output_row)

        # Supernet wird auf den Trainingsdaten ALLER Gruppen trainiert (alt)
        if gruppe == "Supernet":
            N: int = len(training)
            n_klasse: int = 400

            for n in [3, 4, 5, 6]:
                for i in range(3*n_klasse):
                    output_row = [0 for _ in range(len(labels))]
                    output_row[i//n_klasse] = 1
                    output.append(output_row)
                    bag = [0 for _ in range(len(training[0]))]
                    idx = 0
                    idx_list: List = [] # kann doch eigentlich weg, oder??
                    j_list: List[int] = []
                    while sum(bag) < n and idx not in idx_list:
                        idx = torch.randint(N, (1,)).item()
                        if output[idx].index(1) == i//n_klasse:
                            for j, entry in enumerate(training[idx]):
                                if entry == 1:
                                    j_list.append(j)
                                    bag[j] = 1
                    with open("j_list.txt", "a") as file:
                        file.write(str(j_list)+"\n")
                    training.append(bag)

        training = torch.tensor(training).float().to(self.device)
        output = torch.tensor(output).float().to(self.device)

        training = torch.cat((training, torch.zeros(
            int(len(training)/3), training.shape[1]).to(self.device)))
        output = torch.cat((output, torch.zeros(int(len(output)/3), 3).to(self.device)))
        print(training.shape, output.shape)

        # Initialisiere MLP (siehe utils.py) und Optimizer (hier Adam)
        model: Classifier = Classifier([len(words), int(len(words)/2), len(labels)]).to(self.device)
        print(model)
        optimizer: optim.Adam = optim.Adam(model.parameters(), lr=1e-3)

        # Training
        loss_func = F.cross_entropy
        # Meistens VIEL zu viele Epochen, kann früher abgebrochen werden, wenn sich nicht mehr viel ändert (oft weniger als 400)
        n_epochs: int = 10000
        lossliste: Tensor = torch.zeros(n_epochs)

        for epoch in trange(n_epochs):
            # Standard-Trainingsloop
            optimizer.zero_grad()
            out = model(training)
            loss: Tensor = loss_func(out, output)
            loss.backward()
            optimizer.step()
            lossliste[epoch] = loss.item()
            if epoch % int(n_epochs/10) == 0:
                print(epoch, loss.item())
                print(torch.sum(torch.argmax(out, dim=-1) == output @
                    torch.tensor([0., 1., 2.],).to(self.device)).item()/len(output))
        model.eval()

        # Plotte Auswertungen
        plt.figure()
        plt.plot(lossliste.cpu().numpy())
        plt.title(f"Losses {gruppe}")
        plt.grid()
        plt.savefig(f"{Trainer.outputdir}plots/{gruppe}_loss.png")
        plt.figure()
        plt.plot(torch.log(lossliste).cpu().numpy())
        plt.title(f"Loglosses {gruppe}")
        plt.grid()
        plt.savefig(f"{Trainer.outputdir}plots/{gruppe}_logloss.png")

        # Speichere Model und state_dict
        torch.save(model, f"{Trainer.outputdir}/networks/{gruppe}_model.pt")
        torch.save(model.state_dict(),
                f"{Trainer.outputdir}/networks/state_dicts/{gruppe}.pt")
