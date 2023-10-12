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
group_list_dropdown = [name.split(".")[0]
                       for name in os.listdir(inputdir) if "json" in name]

chatbot_option = st.selectbox(
    "ChatBot Auswahl",
    group_list_dropdown,
)
    
with open(f"{inputdir}{chatbot_option}.json", "rb") as file:
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
    with open(f"{state_dicts_path}{chatbot_option}.pt", "rb") as file:
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
