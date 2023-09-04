import streamlit as st
import json
import jsonschema
from jsonschema import validate

#input_path = "/tmp/inputs/"
input_path = "inputs/"

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

def get_list_of_words(words):
    output = words.split("\n")
    output = [word.strip() for word in output]
    output = list(filter(None, output))
    return output

def get_dict(tag, patterns):
    json_dict = {
        "tag": tag,
        "patterns": patterns,
        "responses": [],
        "context_set": "",
    }
    return json_dict

def submit():
    st.write("Daten wurden abgeschickt.")
    st.session_state.isSubmitted = True
    st.session_state.group_name = uploaded_file.name.split(".")[0]
    st.session_state.good_words = data["intents"][0]["patterns"]
    st.session_state.neutral_words = data["intents"][1]["patterns"]
    st.session_state.bad_words = data["intents"][2]["patterns"]

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

tab1, tab2 = st.tabs(["Eingabe", "Upload"])

with tab1:
    with st.form("data_for_own_chatbot", clear_on_submit=False):
        group_name = st.text_input("Gruppenname")
        st.markdown("---")
        good_words = st.text_area("Gute Wörter", value="Mir geht es gut")
        st.markdown("---")
        neutral_words = st.text_area("Neutrale Wörter", value="so lala")
        st.markdown("---")
        bad_words = st.text_area("Schlechte Wörter", value="Mir geht es schlecht")
        st.markdown("---")
        submit = st.form_submit_button("Abschicken")

    if submit:
        group_name = group_name.strip()
        if (not group_name) or (not group_name.isalpha()):
            st.error("ERROR: Vergebt bitte einen Gruppennamen, der aus mindestens einem Zeichen und nur Buchstaben (ohne Sonderzeichen, Zahlen, Leerzeichen) besteht.")
            
        else:
            list_good = get_list_of_words(good_words)
            list_neutral = get_list_of_words(neutral_words)
            list_bad = get_list_of_words(bad_words)

            good_dict = get_dict("good words", list_good)
            neutral_dict = get_dict("neutral words", list_neutral)
            bad_dict = get_dict("bad words", list_bad)

            intents_dict = {
                "intents": [good_dict, neutral_dict, bad_dict],
            }

            with open(f"{input_path}{group_name}.json", "w") as file:
                json.dump(intents_dict, file)
            
            st.write("Daten wurden abgeschickt.")

with tab2:
    if 'blockUpload' not in st.session_state:
        st.session_state.blockUpload = True

    #with st.form("upload_for_own_chatbot", clear_on_submit=False):
    st.write("### Upload")
    #group_name = st.text_input("Gruppenname")
    #st.markdown("---")
    st.write("Lade eine Datei hoch, die die Daten enthält. Der Name der Datei wird als Gruppenname verwendet.")
    uploaded_file = st.file_uploader("Datei hochladen", type=["json"])
    if uploaded_file is not None:
        group_name_upload = uploaded_file.name.split(".")[0].strip()
        st.write("Der Gruppenname ist: ", group_name_upload)
        if (not group_name_upload) or (not group_name_upload.isalpha()):
            st.error("ERROR: Vergebt bitte einen Gruppennamen, der aus mindestens einem Zeichen und nur Buchstaben (ohne Sonderzeichen, Zahlen, Leerzeichen) besteht.")
            st.session_state.blockUpload = True
        
        else:
            st.write("Die Daten aus der hochgeladenen Datei werden hier zur Überprüfung angezeigt. Klickt dafür einfach auf den kleinen Pfeil. Wenn Ihr zufrieden seid, könnt Ihr sie abschicken.")
            data = json.load(uploaded_file)
            st.json(data, expanded=False)
            try:
                validate(instance=data, schema=inputSchema)
                st.session_state.blockUpload = False
                
            except jsonschema.exceptions.ValidationError as err:
                st.error("ERROR: Die Daten entsprechen nicht den Vorgaben. Bitte überprüft sie noch einmal.")
                st.error(err)
                st.session_state.blockUpload = True
                #raise err
    st.markdown("---")
    st.write("### Abschicken")
    st.write("Wenn ihr mit den Daten zufrieden seid, könnt ihr sie abschicken.")
    submit = st.button("Abschicken", disabled=st.session_state.blockUpload)
    if submit:
        with open(f"{input_path}{group_name_upload}.json", "wb") as file:
            file.write(uploaded_file.getbuffer())
        st.write("Daten wurden abgeschickt.")
