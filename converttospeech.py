from gtts import gTTS
import streamlit as st

def convert_text_to_audio(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        audio_path = 'output.mp3'
        tts.save(audio_path)
        return audio_path
    except Exception as E:
        st.error(f'An error occured while converting text ti speech: {E}')