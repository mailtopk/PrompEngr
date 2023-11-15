import streamlit as st
from NlpToSqlPrompt import displayUI
from prompt_custom_data import *

st.set_page_config(page_title="Prompting", page_icon=":robot_face:", layout="wide")


st.sidebar.write('## Prompt Engineering')

prompt_type = st.sidebar.selectbox("Select Option", ["w/ Prompt", "w/o Prompt", "NLP to SQL" ])


if prompt_type == "NLP to SQL":
    displayUI()
if prompt_type == "w/ Prompt":
    CustomPrompt().display_prompt_ui()