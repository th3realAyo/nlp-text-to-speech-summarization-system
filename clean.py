import re
import nltk
from num2words import num2words
from nltk.tokenize import sent_tokenize, word_tokenize

class TTSPreprocessor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Common abbreviations dictionary
        self.abbreviations = {
            'Dr.': 'Doctor',
            'Mr.': 'Mister',
            'Mrs.': 'Misses',
            'Ms.': 'Miss',
            'Prof.': 'Professor',
            'etc.': 'etcetera',
            'vs.': 'versus',
            'e.g.': 'for example',
            'i.e.': 'that is',
        }
        
        # Common homograph dictionary with pronunciation context
        self.homographs = {
            'lead': {'verb': 'LEED', 'noun': 'LED'},
            'read': {'present': 'REED', 'past': 'RED'},
            'wind': {'verb': 'WAYND', 'noun': 'WIND'},
        }

    def clean_text(self, text):
        """Remove HTML tags and standardize whitespace."""
        # Remove HTML tags
        text = re.sub('<[^<]+?>', '', text)
        # Standardize whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text

    def normalize_numbers(self, text):
        """Convert numbers to words."""
        def replace_number(match):
            try:
                return num2words(float(match.group()))
            except:
                return match.group()

        # Handle basic numbers
        text = re.sub(r'\b\d+\.?\d*\b', replace_number, text)
        
        # Handle currency
        text = re.sub(r'\$(\d+\.?\d*)', lambda x: num2words(float(x.group(1))) + ' dollars', text)
        return text

    def expand_abbreviations(self, text):
        """Expand common abbreviations."""
        for abbr, expansion in self.abbreviations.items():
            text = text.replace(abbr, expansion)
        return text

    def segment_sentences(self, text):
        """Break text into sentences."""
        sentences = sent_tokenize(text)
        return sentences

    def tokenize_words(self, sentence):
        """Break sentence into words."""
        return word_tokenize(sentence)

    def mark_prosody(self, sentence):
        """Add basic prosody marks for emphasis and intonation."""
        # Mark questions
        if sentence.endswith('?'):
            sentence = f'<question>{sentence}</question>'
        
        # Mark exclamations
        elif sentence.endswith('!'):
            sentence = f'<emphasis>{sentence}</emphasis>'
            
        # Mark commas for pauses
        sentence = sentence.replace(',', '<pause>,</pause>')
        
        return sentence

    def process_text(self, text):
        """Apply all preprocessing steps."""
        # Step 1: Clean the text
        text = self.clean_text(text)
        
        # Step 2: Normalize numbers
        text = self.normalize_numbers(text)
        
        # Step 3: Expand abbreviations
        text = self.expand_abbreviations(text)
        
        # Step 4: Segment into sentences
        sentences = self.segment_sentences(text)
        
        # Step 5: Process each sentence
        processed_sentences = []
        for sentence in sentences:
            # Tokenize words
            words = self.tokenize_words(sentence)
            
            # Reconstruct sentence
            sentence = ' '.join(words)
            
            # Add prosody marks
            sentence = self.mark_prosody(sentence)
            
            processed_sentences.append(sentence)
        
        return processed_sentences