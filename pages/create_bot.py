import time
import streamlit as st
import json
import jsonschema
from jsonschema import validate
import tasks
from typing import List, Any

#input_path = "/tmp/inputs/"
input_path: str = "inputs/"
timestampformat: str = "%Y-%m-%d-%H-%M-%S"

inputSchema = {
    "type": "object",
    "properties": {
        "intents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string"},
                    "patterns": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "responses": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "context_set": {"type": "string"}
                },
                "required": ["tag", "patterns", "responses"]
            }
        }
    },
    "required": ["intents"]
}

def get_list_of_words(words: str) -> List[str]:
    output: List[str] = [word.strip() for word in words.split("\n")]
    output: List[str] = list(filter(None, output))
    return output

def get_dict(tag: str, patterns: List[str]) -> dict[str, Any]:
    json_dict: dict[str, Any] = {
        "tag": tag,
        "patterns": patterns,
        "responses": [],
        "context_set": "",
    }
    return json_dict


st.write("## 4. Create your own Chatbot")


st.markdown(
    """Vergebt zunächst bitte einen Gruppennamen, der aus mindestens einem Zeichen und nur Buchstaben (ohne Sonderzeichen, Zahlen, Leerzeichen) besteht. 
    Anschließend könnt ihr euch "gute", "neutrale" sowie "schlechte" Wörter überlegen und in die dafür vorgesehenen Bereichen eintragen. 
    Beachtet dabei, dass ihr einzelne Wörter (z.B. "gut" als gutes Wort) oder auch zusammengehörende Wörter (z.B. "sehr gut" als gutes Wort) jeweils in einzelne Zeilen schreiben und dann eine neue Zeile beginnt. 
    Zum Schluss, wenn ihr fertig seid, klickt auf "Abgeben"."""
)
st.markdown(
    """Der Chatbot erhält beim Training eure gelabelten Daten. Ziel ist es, dass er hinterher möglichst gut auf Nutzereingaben reagieren kann. Es sollte bei den Daten also darauf geachtet werden, dass möglichst "natürliche" Eingaben gegeben werden sollten.
    **Hinweis**: Die einzelnen Eingaben müssen durch Zeilenumbrüche getrennt sein."""
)

tab_input, tab_upload = st.tabs(["Eingabe", "Upload"])

with tab_input:
    with st.form("data_for_own_chatbot", clear_on_submit=False):
        group_name: str = st.text_input("Gruppenname")
        st.markdown("---")
        good_words: str = st.text_area("Gute Wörter", value="Mir geht es gut")
        st.markdown("---")
        neutral_words: str = st.text_area("Neutrale Wörter", value="so lala")
        st.markdown("---")
        bad_words: str = st.text_area("Schlechte Wörter", value="Mir geht es schlecht")
        st.markdown("---")
        distilbert: bool = st.checkbox("DistilBERT verwenden", value=False)
        st.markdown("---")
        submit: bool = st.form_submit_button("Abschicken")

    if submit:
        group_name: str = group_name.strip()
        if (not group_name) or (not group_name.isalnum()):
            st.error("ERROR: Vergebt bitte einen Gruppennamen, der aus mindestens einem Zeichen und nur Buchstaben (ohne Sonderzeichen, Zahlen, Leerzeichen) besteht.")
            
        else:
            list_good: List[str] = get_list_of_words(good_words)
            list_neutral: List[str] = get_list_of_words(neutral_words)
            list_bad: List[str] = get_list_of_words(bad_words)

            good_dict: dict[str, Any] = get_dict("good words", list_good)
            neutral_dict: dict[str, Any] = get_dict("neutral words", list_neutral)
            bad_dict: dict[str, Any] = get_dict("bad words", list_bad)

            intents_dict: dict[str, list[dict[str, Any]]] = {
                "intents": [good_dict, neutral_dict, bad_dict],
            }

            # human readable timestamp
            timestamp: str = time.strftime(timestampformat)
            if distilbert:
                timestamp += "-distilBERT"

            with open(f"{input_path}{group_name}_{timestamp}.json", "w") as file:
                json.dump(intents_dict, file, ensure_ascii=False, indent=4)
            
            st.write("Daten wurden abgeschickt.")

            # offer file for download
            with open(f"{input_path}{group_name}_{timestamp}.json", "rb") as file:
                fileContent: bytes = file.read()
                st.download_button(
                    label="Download",
                    data=fileContent,
                    file_name=f"{group_name}.json",
                    mime="application/json",
                )

            if distilbert:
                tasks.train_distilbert.delay(group_name, timestamp)
            else:
                tasks.train.delay(group_name, timestamp, stemmer="cistem")

with tab_upload:
    if 'blockUpload' not in st.session_state:
        st.session_state.blockUpload = True

    st.write("## Template")
    st.write("Hier könnt ihr euch ein Template herunterladen, um eure Daten in das richtige Format zu bringen. Bitte achtet darauf, die Datei in Eurem Gruppennamen umzubenennen.")
    with open("template.json", "rb") as file:
        fileContent: bytes = file.read()
        st.download_button(
            label="Download Template",
            data=fileContent,
            file_name="template.json",
            mime="application/json",
        )

    #with st.form("upload_for_own_chatbot", clear_on_submit=False):
    st.write("### Upload")
    #group_name = st.text_input("Gruppenname")
    #st.markdown("---")
    st.write("Lade eine Datei hoch, die die Daten enthält. Der Name der Datei wird als Gruppenname verwendet.")
    uploaded_file = st.file_uploader("Datei hochladen", type=["json"])
    group_name_upload: str = ""
    if uploaded_file is not None:
        group_name_upload: str = uploaded_file.name.split(".")[0].strip()
        st.write("Der Gruppenname ist: ", group_name_upload)
        if (not group_name_upload) or (not group_name_upload.isalnum()):
            st.error("ERROR: Vergebt bitte einen Gruppennamen, der aus mindestens einem Zeichen und nur Buchstaben (ohne Sonderzeichen, Zahlen, Leerzeichen) besteht.")
            st.session_state.blockUpload = True
        
        else:
            st.write("Die Daten aus der hochgeladenen Datei werden hier zur Überprüfung angezeigt. Klickt dafür einfach auf den kleinen Pfeil. Wenn Ihr zufrieden seid, könnt Ihr sie abschicken.")
            data = json.load(uploaded_file)
            st.json(data, expanded=False)
            try:
                validate(instance=data, schema=inputSchema)
                st.session_state.blockUpload = False
                
            except jsonschema.ValidationError as err:
                st.error("ERROR: Die Daten entsprechen nicht den Vorgaben. Bitte überprüft sie noch einmal.")
                st.error(err)
                st.session_state.blockUpload = True
                #raise err
    st.markdown("---")
    distilbert: bool = st.checkbox("DistilBERT verwenden", value=False)
    st.markdown("---")
    st.write("### Abschicken")
    st.write("Wenn ihr mit den Daten zufrieden seid, könnt ihr sie abschicken.")
    submit: bool = st.button("Abschicken", disabled=st.session_state.blockUpload)
    if submit:

        if uploaded_file is not None:

            timestamp: str = time.strftime(timestampformat)
            if distilbert:
                timestamp += "-distilBERT"

            with open(f"{input_path}{group_name_upload}_{timestamp}.json", "wb") as file:
                file.write(uploaded_file.getbuffer())
            st.write("Daten wurden abgeschickt.")

            if distilbert:
                tasks.train_distilbert.delay(group_name_upload, timestamp)
            else:
                tasks.train.delay(group_name_upload, timestamp, stemmer="cistem")
        else:
            st.error("ERROR: Es wurde keine Datei hochgeladen.")

    
