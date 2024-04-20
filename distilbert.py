# Load model directly
import os
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '.cache/huggingface/hub'
os.environ['HUGGINGFACE_HUB_CACHE'] = '.cache/huggingface/hub'
os.environ['HF_HOME'] = '.cache/huggingface/hub'
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
torch.hub.set_dir(".cache")
from utils import Dataset, Classifier
from tqdm.auto import tqdm
from math import cos
from torch import Tensor
import streamlit as st

class Distilbert:

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-german-cased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-german-cased")
    model.pre_classifier = nn.Identity()
    model.classifier = nn.Identity()

    def train(self, groupname: str, timestamp: str):

        classifier = Classifier([768, 768//2, 3]) #nn.Linear(768,3)
        print(classifier)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

        D = Dataset(self.model, self.tokenizer, groupname, timestamp)
        D_test = Dataset(self.model, self.tokenizer, groupname, timestamp, train=False)
        batch_size = 16
        dataloader = torch.utils.data.DataLoader(D, batch_size=batch_size, shuffle=True)
        dataloader_test = torch.utils.data.DataLoader(D_test, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss(reduction="sum")

        n_epochs = 1_000
        lossliste: Tensor = torch.zeros(n_epochs)
        bar = tqdm(range(n_epochs))
        lr_start = 1e-3
        lr_end = 1e-5
        for epoch in bar:
            lr = lr_end + (lr_start - lr_end) * (1 + cos(epoch/n_epochs))/2
            classifier.train()
            l, n, a = 0, 0, 0
            for batch, label in dataloader:
                optimizer.zero_grad()
                output = classifier(batch)
                loss = criterion(output, label.long())
                acc = (output.argmax(1) == label).float().sum()
                a += acc.item()
                l += loss.item()
                n += len(label)
                lossliste[epoch] = loss.item()
                
                loss.backward()
                optimizer.step()
            l_train = l/n
            a_train = a/n

            if epoch % 10 == 0:
                l, n, a = 0, 0, 0
                for batch, label in dataloader_test:
                    output = classifier(batch)
                    loss = criterion(output, label.long())
                    acc = (output.argmax(1) == label).float().sum()
                    a += acc.item()
                    l += loss.item()
                    n += len(label)
                l_test = l/n
                a_test = a/n
                bar.set_description(f"[{epoch+1}/{n_epochs}] L_train: {l_train:.2e}, L_test: {l_test:.2e}, Acc_train: {a_train*100:.2f}%, Acc_test: {a_test*100:.2f}%, lr: {lr:.2e}")

        torch.save(classifier, f"outputs/networks/{groupname}_{timestamp}_model.pt")
        torch.save(classifier.state_dict(), f"outputs/networks/state_dicts/{groupname}_{timestamp}.pt")
        torch.save(lossliste, f"outputs/losses/{groupname}_{timestamp}_losses.pt")

        # Plotte Auswertungen
        plt.figure()
        plt.plot(lossliste.cpu().numpy())
        plt.title(f"Losses {groupname} at {timestamp}")
        plt.grid()
        plt.savefig(f"outputs/plots/{groupname}_{timestamp}_loss.png")
        plt.figure()
        plt.plot(torch.log(lossliste).cpu().numpy())
        plt.title(f"Loglosses {groupname} at {timestamp}")
        plt.grid()
        plt.savefig(f"outputs/plots/{groupname}_{timestamp}_logloss.png")

        # eingabe = ""
        # labeldict = {0: "good words", 1: "bad words", 2: "neutral words"}
        # while eingabe!="exit":
        #     eingabe = input("Eingabe: ")
        #     ttt = tokenizer([eingabe], return_tensors="pt", padding=True, truncation=True)
        #     out = (classifier(model(ttt.input_ids, attention_mask=ttt.attention_mask).logits))
        #     print(out)
        #     print(labeldict[out.argmax().item()])

        #print(model)
        #print(sum([p.numel() for p in model.parameters()]))
        #print(tokenizer)
