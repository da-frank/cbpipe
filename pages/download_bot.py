import streamlit as st
import requests
import os

def download_file(url, file_name):
    response = requests.get(url)
    with open(file_name, "wb") as f:
        f.write(response.content)
    st.success(f"{file_name} downloaded successfully!")


inputdir = "inputs/"
state_dicts_path = "outputs/networks/state_dicts/"

st.title("Download Chatbot Data")
# get list of groups from inputs directory
group_list_dropdown = ["".join(name.split("_")[:-1])
                       for name in os.listdir(inputdir) if "json" in name]

if len(group_list_dropdown) == 0:
    st.write("Kein ChatBot vorhanden. Bitte erstelle einen ChatBot.")
    st.stop()

chatbot_option = st.selectbox(
    "ChatBot auswählen",
    group_list_dropdown,
)

timestamp = st.selectbox(
    "Version auswählen",
    [name.split("_")[-1].split(".")[0]
     for name in os.listdir(inputdir) if chatbot_option in name],
)
    
with open(f"{inputdir}{chatbot_option}_{timestamp}.json", "rb") as file:
    fileContentJSON: bytes = file.read()
    jsonAvailable = True

st.download_button(
    label="Download JSON",
    data=fileContentJSON,
    disabled=not jsonAvailable,
    file_name=f"{chatbot_option}.json",
    mime="application/json",
)

try:
    with open(f"{state_dicts_path}{chatbot_option}_{timestamp}.pt", "rb") as file:
        fileContentPT: bytes = file.read()
        ptAvailable = True
except FileNotFoundError:
    fileContentPT = fileContentJSON
    ptAvailable = False

st.download_button(
    label="Download State Dict",
    data=fileContentPT,
    disabled=not ptAvailable,
    file_name=f"{chatbot_option}.pt",
    mime="application/pt",
)
