# Text-to-Audio Summarizer (Final Year Project)

## Mini Intro

This project is an NLP-based system developed as part of my undergraduate final year project. It allows users to upload PDF or Word documents, automatically summarizes the content using cosine similarity, and generates an audio version of the summary in the document’s original language.
The goal is to help students quickly understand large volumes of text through concise summaries and audio playback.

---

## Features

* Upload and process PDF/Word documents
* Text summarization using cosine similarity
* Text-to-Speech (TTS) audio generation
* Language detection and translation support
* Efficient handling of large documents

---

## Libraries

The project uses the following Python libraries:

* `nltk` – Natural Language Processing
* `num2words` – Converts numbers to words
* `pdfplumber` – Extract text from PDFs
* `PyPDF2` – PDF processing
* `pytesseract` – OCR for scanned documents
* `Pillow` – Image processing
* `langdetect` – Language detection
* `googletrans==3.1.0a0` – Translation support
* `gTTS` – Text-to-Speech conversion
* `transformers` – Advanced NLP models
* `networkx` – Graph-based ranking (for summarization)
* `scipy` – Cosine similarity calculations
* `ipykernel` – Jupyter Notebook support

---

## How It Works

1. User uploads a document (PDF or Word)
2. Text is extracted from the file
3. If needed, OCR is applied to scanned documents
4. Language of the text is detected
5. Sentences are ranked using cosine similarity
6. A summary is generated
7. The summary is converted to speech using TTS
8. Audio output is returned to the user

---

## Installation

```bash
git clone https://github.com/th3realAyo/nlp-text-to-speech-summarization-system.git
cd nlp-text-to-speech-summarization-system
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

---

## Future Improvements

* Improve summarization using transformer-based models
* Add real-time streaming audio output
* Build a web/mobile interface
* Support more file formats

---

## Author
Adegunte Damilola
