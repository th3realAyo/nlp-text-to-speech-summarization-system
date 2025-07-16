import streamlit as st
import pdfplumber as pf
from clean import TTSPreprocessor
from summarize import TextSummarizer, MultilingualProcessor
from converttospeech import convert_text_to_audio



processor = TTSPreprocessor()
multi_processor = MultilingualProcessor()
english_summarizer = TextSummarizer()

def extract_text(pdf_file):
    '''
    This function helps extract pdf content, especially text from a given pdf file.
    '''

    try:
        extract_text = ""
        with pf.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                print(f'Processing for page {page_num}'.upper())
                print('====' * 20)
                text = page.extract_text()
                if text:
                    extract_text += text
        return extract_text
    except Exception as E:
        st.error(f'An error occured proccessing: {E}')
        return str(E)
        
def main():
    st.title("PDF Text to Audio Converter")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.success('File Uploaded Successfully')
        
        with st.spinner("Extracting text from PDF..."):
            text = extract_text(uploaded_file)

        if text:
            with st.spinner("Processing text..."):
                
                lang_code, lang_name = multi_processor.detect_language(text)
                # st.write(lang_name)
    

                if lang_code == 'en':
                    text = processor.process_text(text)
                    text = english_summarizer.extractive_summarize(text)
                else:
                    text = multi_processor.summarize_multilingual(text)

            st.text_area('Extracted and Summarized Text', value=text, height=300)

            if st.button('Convert to Audio'):
                with st.spinner("Converting text to audio..."):
                    audio_file_path = convert_text_to_audio(text, lang=lang_code)
                
                st.write('Detected Language:', lang_name)
                
                if audio_file_path:
                    st.audio(audio_file_path, format='audio/mp3')
                    st.success("Audio conversion completed successfully!")
                    with open(audio_file_path, "rb") as file:
                        btn = st.download_button(
                            label="Download Audio",
                            data=file,
                            file_name="output.mp3",
                            mime="audio/mp3",
                        )
        else:
            st.warning('No text found in the PDF.')


if __name__ == '__main__':
    main()

