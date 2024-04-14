import numpy as np
import streamlit as st
import nltk
import os
from random import randint

import pandas as pd
import torch

from models import Netpicker

# from utils import Classifier
# import matplotlib

inputdir = "inputs/"


def load_token():
    with open("token.txt", "r") as file:
        t = file.read()
    t = t.split("/")
    return t[randint(0, len(t))]


@st.cache(suppress_st_warning=True)
def download_punkt():
    nltk.download("punkt")

# ---------------------------------------------------------------------------------------------------------


st.markdown("## Vergleich der ChatBots")

st.markdown(
    "Hier können die ChatBots verschiedener Gruppen geladen und getestet werden."
)

st.markdown("---")

# get list of groups from inputs directory
group_list_dropdown = [name.split(".")[0]
                       for name in os.listdir(inputdir) if "json" in name]

chatbot_option = st.selectbox(
    "ChatBot Auswahl",
    group_list_dropdown,
)

# TODO take Gruppe(...) instead of Testgruppe
print("Loading", chatbot_option)
current_group = Netpicker(chatbot_option)

table_current_group_input = chatbot_option + "user_table_entries"
table_current_group_good = chatbot_option + "good_table_entries"
table_current_group_bad = chatbot_option + "bad_table_entries"
table_current_group_neutral = chatbot_option + "neutral_table_entries"
table_current_group_entropy = chatbot_option + "entropy_table_entries"
if table_current_group_input not in st.session_state:
    # st.write("Group not in session state")
    st.session_state[table_current_group_input] = []
    st.session_state[table_current_group_good] = []
    st.session_state[table_current_group_bad] = []
    st.session_state[table_current_group_neutral] = []
    st.session_state[table_current_group_entropy] = []

st.markdown("---")

st.markdown("Chatbot ("+current_group.name+"): Wie geht es dir heute?")

with st.form("user_input", clear_on_submit=True):
    user_input = st.text_input("Nutzer:", key="input_sentence")
    submit = st.form_submit_button(label="Senden")

st.markdown("---")

if submit:
    result = current_group.predict(user_input)
    #calculate shannon entropy of result
    shannon_entropy = -sum([p.item() * np.log2(p.item()) for p in result])

    st.session_state[table_current_group_input].append(user_input)
    st.session_state[table_current_group_good].append(result[1].item())
    st.session_state[table_current_group_bad].append(result[0].item())
    st.session_state[table_current_group_neutral].append(result[2].item())
    st.session_state[table_current_group_entropy].append(shannon_entropy)

    st.markdown("Details zu Antwort")
    result_table = {
        "Nutzer": st.session_state[table_current_group_input],
        "gut": st.session_state[table_current_group_good],
        "schlecht": st.session_state[table_current_group_bad],
        "neutral": st.session_state[table_current_group_neutral],
        "Shannon Entropy": st.session_state[table_current_group_entropy]
    }

    result_table = pd.DataFrame.from_dict(result_table)

    st.table(result_table.style.background_gradient(axis=None, cmap="Blues"))
    st.markdown("---")
    st.markdown(f"Überprüfungsschlüssel zur Abgabe: **{load_token()}**")

    # st.sidebar.image("./images/Logo_Uni_Luebeck_600dpi.png", use_column_width=True)
    # st.sidebar.image("./images/Logo_UKT.png", use_column_width=True)

st.markdown("---")

st.markdown("### Performce des Bots")

# hier die performancewerte anzeigen
# show plot of loss and logloss
st.image([f"outputs/plots/{current_group.name}_loss.png",
          f"outputs/plots/{current_group.name}_logloss.png"],
         caption=["loss", "log loss"], use_column_width="auto")

# show table of losses
st.markdown("### Losses")
losses = torch.load(f"outputs/losses/{current_group.name}_losses.pt")
st.line_chart(losses.numpy())