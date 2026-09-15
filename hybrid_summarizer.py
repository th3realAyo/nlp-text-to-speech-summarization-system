"""
Hybrid summarization module.

Strategy:
1. Extractive pre-filter (TextRank) trims a long document down to its most
   relevant sentences. This is cheap, has no model-size limits, and avoids
   silently truncating important content before it ever reaches the
   abstractive model.
2. Abstractive rewrite (distilbart-cnn-12-6) turns the extracted sentences
   into a fluent, non-choppy summary. It's a distilled model, so it runs at
   reasonable speed on CPU-only hosting (e.g. a free Hugging Face Space).

Long inputs are chunked to stay within the abstractive model's token limit,
each chunk is summarized independently, and (if the combined result is
still long) a second pass compresses it further.
"""

from typing import List

from transformers import AutoTokenizer, pipeline

from textrank_summarizer import TextRankSummarizer

# Distilled BART checkpoint: much lighter than facebook/bart-large-cnn,
# CPU-friendly, still solid quality for news/article-style text.
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# Leave headroom under the model's real limit (1024) for special tokens.
MAX_INPUT_TOKENS = 900


class HybridSummarizer:
    def __init__(self, model_name: str = MODEL_NAME):
        self.textrank = TextRankSummarizer()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=self.tokenizer,
            device=-1,  # force CPU; set to 0 if a GPU Space is ever used
        )

    def _chunk_by_tokens(self, text: str, max_tokens: int = MAX_INPUT_TOKENS) -> List[str]:
        """Split text into chunks that each fit within the model's token limit."""
        sentences = text.replace("\n", " ").split(". ")
        chunks, current, current_len = [], [], 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_len = len(self.tokenizer.encode(sentence, add_special_tokens=False))

            if current_len + sentence_len > max_tokens and current:
                chunks.append(". ".join(current) + ".")
                current, current_len = [], 0

            current.append(sentence)
            current_len += sentence_len

        if current:
            chunks.append(". ".join(current) + ".")

        return chunks

    def _abstractive_pass(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        chunks = self._chunk_by_tokens(text)
        summaries = []

        for chunk in chunks:
            input_len = len(self.tokenizer.encode(chunk, add_special_tokens=False))
            if input_len < 10:
                # Too short to meaningfully summarize; keep as-is.
                summaries.append(chunk)
                continue

            result = self.summarizer(
                chunk,
                max_length=max_length,
                min_length=min(min_length, max(5, input_len // 2)),
                do_sample=False,
                truncation=True,
            )
            summaries.append(result[0]["summary_text"].strip())

        return " ".join(summaries)

    def summarize(
        self,
        text: str,
        prefilter_sentences: int = 15,
        max_length: int = 130,
        min_length: int = 30,
    ) -> str:
        """
        Produce a hybrid extractive + abstractive summary.

        Args:
            text: Raw input text (e.g. extracted PDF text).
            prefilter_sentences: How many sentences TextRank should keep
                before handing off to the abstractive model.
            max_length: Max token length per abstractive summary chunk.
            min_length: Min token length per abstractive summary chunk.
        """
        if not text or not text.strip():
            return ""

        # Step 1: extractive pre-filter to cut down noise/length cheaply.
        prefiltered = self.textrank.summarize(text, num_sentences=prefilter_sentences)

        # Step 2: abstractive rewrite, chunked to respect token limits.
        summary = self._abstractive_pass(prefiltered, max_length=max_length, min_length=min_length)

        # Step 3: if multiple chunks produced a long combined summary,
        # compress once more into a single coherent pass.
        combined_len = len(self.tokenizer.encode(summary, add_special_tokens=False))
        if combined_len > max_length * 1.5:
            summary = self._abstractive_pass(summary, max_length=max_length, min_length=min_length)

        return summary
