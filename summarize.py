from langdetect import detect
from googletrans import Translator
from gtts import gTTS
from transformers import MarianMTModel, MarianTokenizer
import os
import json



import re
import nltk
from num2words import num2words
from nltk.tokenize import sent_tokenize, word_tokenize
# from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from collections import defaultdict
import networkx as nx

nltk.download('punkt_tab')

class TextSummarizer:
    def __init__(self):
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        self.stopwords = set(stopwords.words('english'))

    def clean_process_texts(self, processed_txt):
        """
        Clean processed texts, removing prosody
        """
        cleaned_sentences = []
        for sentence in processed_txt:
            cleaned = re.sub(r'<[^>]+?>', '', sentence)
            cleaned_sentences.append(cleaned)
        return cleaned_sentences
        
    def create_sentence_vector(self, sentences):
        word_frequency = defaultdict(int)

        #First Pass: count all words
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            for word in words:
                if word not in self.stopwords and word.isalnum():
                    word_frequency[word] += 1
                    
        sentence_vectors = []
        for sentence in sentences:
            vector = []
            words = word_tokenize(sentence.lower())
            for word in word_frequency.keys():
                if word in words:
                    vector.append(word_frequency[word])
                else: vector.append(0)
            sentence_vectors.append(vector)
        return sentence_vectors

    def calculate_cosine_similarity(self, vect1, vect2):
        """Calculate cosine similarity with proper handling of zero vectors"""
        if not any(vect1) or not any(vect2):
            return 0.0
        dot_product = np.dot(vect1, vect2)
        norm1 = np.sqrt(np.dot(vect1, vect1))
        norm2 = np.sqrt(np.dot(vect2, vect2))

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def similiarity_matrix(self, sentence_vectors):
        """
        Using Cosine Similarity, we build a matrix
        """
        similarity_matrix = np.zeros((len(sentence_vectors), len(sentence_vectors)))

        for i in range(len(sentence_vectors)):
            for j in range(len(sentence_vectors)):
                if i != j:
                    
                    similarity_matrix[i][j] = self.calculate_cosine_similarity(
                        sentence_vectors[i], sentence_vectors[j]
                    )
        return similarity_matrix

    def extractive_summarize(self, processed_sentences, num_sentences=3):
        """
        Generate extractive summaryusing TextRankalgorithm
        """
        cleaned_sentences = self.clean_process_texts(processed_sentences)
        if len(cleaned_sentences) <= num_sentences:
            return cleaned_sentences

        # Create sentence vectors
        sentence_vectors = self.create_sentence_vector(cleaned_sentences)
        #Build similarity matrix
        similarity_matrix = self.similiarity_matrix(sentence_vectors)

        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

        # select top sentences
        ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(cleaned_sentences)), reverse=True)
        # Get the top n sentences while preserving original order
        selected_indices = sorted([cleaned_sentences.index(sentence) 
                                 for score, sentence in ranked_sentences[:num_sentences]])
        
        summary = [cleaned_sentences[i] for i in selected_indices]

        return ''.join(summary)
                


class MultilingualProcessor:
    def __init__(self):
        self.translator = Translator()

        #Create a dictionary tagging language codes to their full names
        self.language_codes = {
            'en' : 'english',
            'es' : 'spanish',
            'fr' : 'french',
            'de' : 'german',
            'it' : 'italian',
            'pt' : 'portugese',
            'nl' : 'dutch',
            'ru' : 'russian',
            'ar' : 'arabic',
            'zh' : 'chinese',
            'ja' : 'japanese',
            'ko' : 'korean'
        }

        for lang in self.language_codes.values():
            try:
                nltk.data.find(f'tokenizers/punkt/{lang}.pickle')
            except LookupError:
                nltk.download('punkt')
    def detect_language(self, text):
        '''
        Detect the language of the text
        '''

        try:
            lang_code = detect(text)
            return lang_code, self.language_codes.get(lang_code, 'unknown')
        except:
            return 'en', 'english'
        
    def summarize_multilingual(self, text, target_length=0.3):
        '''
        Summarizes text in its original language
        '''
        self.lang_code, lang_name = self.detect_language(text)
        # print(lang_name)

        #Tokenize text based on language
        sentences = nltk.sent_tokenize(text, language=lang_name)

        if not sentences:
            return text
        #let's calculate word frequences
        words = nltk.word_tokenize(text.lower())
        word_freq = {}
        for word in words:
            if word.isalnum():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # calculate sentence scores

        sentence_scores = {

        }
        for sentence in sentences:
            for word in nltk.word_tokenize(sentence.lower()):
                if word in word_freq:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_freq[word]
                    else:
                        sentence_scores[sentence] += word_freq[word]
        
        # select top sentences
        num_sentences = max(1, int(len(sentences) * target_length))
        summary_sentences = sorted(sentence_scores.items(), key=lambda x : x[1], reverse=True)[:num_sentences]

        summary = [s[0] for s in summary_sentences]
        print(summary)
        summary.sort(key=lambda s: sentences.index(s))
        return ' '.join(summary)

    def translate_text(self, text, target_lang='en'):
        '''
        Translate text to target language
        '''
        translation = self.translator.translate(text, dest=target_lang)
        return translation.text
    
    def text_to_speech(self, text, output_path, languge='en'):
        try:
            tts = gTTS(text=text, lang=self.lang_code)
            tts.save(output_path)
            return output_path
        except Exception as E:
            print('TTS Error: ' , str(E))
            return False
    def process_multilingual_content(self, text, output_dir='output'):
        os.makedirs(output_dir, exist_ok=True)
        lang_code, lang_name = self.detect_language(text)

        print(f'Language Detected is: {lang_code} - {lang_name}')

        #Create summary of the PDF text
        if lang_code == 'en' :
            summarizer = TextSummarizer()  
            text = summarizer.extractive_summarize(text)
            original_audio_path = os.path.join(output_dir, f'summary_{lang_code}.mp3')
            self.text_to_speech(text, original_audio_path, lang_code)
        else:
            original_audio_path = os.path.join(output_dir, f'summary_{lang_code}.mp3')
            text = self.summarize_multilingual(text)
            self.text_to_speech(text, original_audio_path, lang_code)
        return original_audio_path
        