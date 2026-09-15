"""
modal_app.py

Modal application hosting the GPU-backed ML inference for the grounded
RAG summarization pipeline.

Exposes two authenticated HTTP endpoints, called by the Streamlit app
(hosted separately on a free Hugging Face Space):

    POST /embed        -> embeds a list of text chunks (BAAI/bge-base-en-v1.5)
    POST /summarize     -> grounded, per-chunk abstractive rewrite
                            (Llama-3.1-8B-Instruct, 4-bit quantized)

A third endpoint for self-hosted TTS fallback (XTTS-v2) will be added
in a later pass once this core RAG loop is confirmed working.

Deploy with:
    modal deploy modal_app.py

Required Modal secrets (create once via `modal secret create ...`):
    hf-token        -> HF_TOKEN            (Hugging Face token with access
                                             to the gated meta-llama/Llama-3.1-8B-Instruct repo)
    rag-auth-token  -> RAG_AUTH_TOKEN      (shared secret the Streamlit app
                                             must send in the Authorization header)
"""

import os
from typing import List

import modal

# --------------------------------------------------------------------------
# App + image definition
# --------------------------------------------------------------------------

app = modal.App("pdf-rag-summarizer")

# Single image shared by both GPU classes. Kept in one image for simplicity;
# split into two images later if cold-start size becomes an issue.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "bitsandbytes==0.43.3",
        "sentence-transformers==3.0.1",
        "fastapi[standard]==0.115.0",
        "pydantic==2.9.2",
    )
)

# Separate image for YarnGPT: it needs its own dependency set (outetts,
# uroman, gdown) plus the yarngpt GitHub repo and WavTokenizer weights baked
# in at build time. Kept isolated from the embed/summarize image so a
# YarnGPT setup issue can't break the core RAG pipeline, and so the
# Embedder/Summarizer containers don't carry unrelated weight.
yarngpt_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .pip_install(
        "torch==2.4.0",
        "torchaudio==2.4.0",
        "transformers==4.46.1",  # outetts 0.2.3 requires >=4.46.1; the 4.44.2
                                  # pin used in the base image conflicts here
        "outetts==0.2.3",
        "uroman",
        "inflect",
        "numpy",
        "gdown",
        "fastapi[standard]==0.115.0",
        "pydantic==2.9.2",
    )
    .run_commands(
        # Clone the yarngpt repo to get its `audiotokenizer` module — it
        # isn't published on PyPI, so this is the model author's documented
        # way to use it (see saheedniyi/YarnGPT2 model card).
        "git clone https://github.com/saheedniyi02/yarngpt.git /root/yarngpt_src",
        # WavTokenizer config + checkpoint, used to encode/decode audio.
        "wget -O /root/wavtokenizer_config.yaml "
        "https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/"
        "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        "gdown 1-ASeEkrn4HY49yZWHTASgfGFNXdVnLTt -O /root/wavtokenizer_model.ckpt",
    )
)

# --------------------------------------------------------------------------
# Auth helper
# --------------------------------------------------------------------------
# Simple shared-secret check so the public Modal endpoint can't be called by
# anyone other than the Streamlit app. Not full OAuth-grade auth, but
# sufficient for a single-frontend internal API.


def _check_auth(authorization: str | None) -> None:
    from fastapi import HTTPException

    expected = os.environ["RAG_AUTH_TOKEN"]
    if not authorization or authorization.removeprefix("Bearer ") != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing auth token")


# --------------------------------------------------------------------------
# Request/response schemas
# --------------------------------------------------------------------------

from pydantic import BaseModel


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


class SummarizeRequest(BaseModel):
    text: str
    max_new_tokens: int = 120
    language: str = "English"  # e.g. "English", "French" — matches detected doc language


class SummarizeResponse(BaseModel):
    summary: str


class TranslateRequest(BaseModel):
    text: str
    target_language: str  # full name, e.g. "English", "Yoruba"
    max_new_tokens: int = 300


class TranslateResponse(BaseModel):
    translated_text: str


class YarnGPTRequest(BaseModel):
    text: str
    language: str  # "english" | "yoruba" | "igbo" | "hausa"
    speaker_name: str  # see YARNGPT_VOICES below for valid values per language


class YarnGPTResponse(BaseModel):
    audio_base64: str  # WAV audio, base64-encoded (JSON has no native binary type)


# --------------------------------------------------------------------------
# Embedding model: BAAI/bge-base-en-v1.5
# --------------------------------------------------------------------------


@app.cls(
    image=image,
    gpu="A10G",
    scaledown_window=120,  # keep container warm for 2 min after last request
)
class Embedder:
    @modal.enter()
    def load_model(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")

    @modal.method()
    def embed(self, texts: List[str]) -> List[List[float]]:
        # bge models recommend a query/passage prefix convention; for pure
        # passage-importance scoring (no separate query), embedding the raw
        # passage text directly is the right call here.
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    @modal.fastapi_endpoint(method="POST")
    def embed_endpoint(self, request: EmbedRequest, authorization: str = None):
        _check_auth(authorization)
        vectors = self.embed.local(request.texts)
        return EmbedResponse(embeddings=vectors)


# --------------------------------------------------------------------------
# Abstractive rewriting model: Llama-3.1-8B-Instruct (4-bit quantized)
# --------------------------------------------------------------------------

GROUNDING_SYSTEM_PROMPT = (
    "You are a precise summarization assistant. You will be given a single "
    "passage extracted from a larger document. Summarize ONLY the "
    "information contained in that passage, in 1-3 concise sentences. "
    "Do not add facts, context, or interpretation that is not explicitly "
    "present in the passage. Do not speculate. Respond in {language}."
)

# Reuses the same loaded Llama-3.1-8B-Instruct model as summarization —
# translation is just a different prompt, not a different model, so this
# adds zero extra GPU memory or cold-start cost on top of the Summarizer
# container that's already running.
TRANSLATION_SYSTEM_PROMPT = (
    "You are a precise translation assistant. Translate the following text "
    "into {language}. Preserve the original meaning exactly — do not "
    "summarize, expand, or add commentary. Respond with ONLY the "
    "translated text, nothing else."
)


@app.cls(
    image=image,
    gpu="A10G",
    scaledown_window=120,
    secrets=[modal.Secret.from_name("hf-token")],
)
class Summarizer:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        hf_token = os.environ["HF_TOKEN"]

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            token=hf_token,
        )

    @modal.method()
    def summarize(self, text: str, max_new_tokens: int = 120, language: str = "English") -> str:
        import torch

        messages = [
            {"role": "system", "content": GROUNDING_SYSTEM_PROMPT.format(language=language)},
            {"role": "user", "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic, favors grounding over creativity
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @modal.fastapi_endpoint(method="POST")
    def summarize_endpoint(self, request: SummarizeRequest, authorization: str = None):
        _check_auth(authorization)
        summary = self.summarize.local(
            request.text, request.max_new_tokens, request.language
        )
        return SummarizeResponse(summary=summary)

    @modal.method()
    def translate(self, text: str, target_language: str, max_new_tokens: int = 300) -> str:
        import torch

        messages = [
            {
                "role": "system",
                "content": TRANSLATION_SYSTEM_PROMPT.format(language=target_language),
            },
            {"role": "user", "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @modal.fastapi_endpoint(method="POST")
    def translate_endpoint(self, request: TranslateRequest, authorization: str = None):
        _check_auth(authorization)
        translated = self.translate.local(
            request.text, request.target_language, request.max_new_tokens
        )
        return TranslateResponse(translated_text=translated)


# --------------------------------------------------------------------------
# YarnGPT: Nigerian-accented TTS (English, Yoruba, Igbo, Hausa)
# --------------------------------------------------------------------------
# Voices as documented on the saheedniyi/YarnGPT2 model card, arranged by
# the author in order of performance/stability. Note: "yoruba_feamle1" and
# "hausa_feamle1" are typo'd in the upstream model itself (not a typo on
# our end) — these exact strings are the real speaker_name values the
# model expects, so they're preserved as-is rather than "corrected."
YARNGPT_VOICES = {
    "english": ["idera", "chinenye", "jude", "emma", "umar", "joke", "zainab", "osagie", "remi", "tayo"],
    "yoruba": ["yoruba_male2", "yoruba_female2", "yoruba_feamle1"],
    "igbo": ["igbo_female2", "igbo_male2", "igbo_female1"],
    "hausa": ["hausa_feamle1", "hausa_female2", "hausa_male2", "hausa_male1"],
}


@app.cls(
    image=yarngpt_image,
    gpu="A10G",
    scaledown_window=120,
)
class YarnGPTSynthesizer:
    @modal.enter()
    def load_model(self):
        import sys

        sys.path.insert(0, "/root/yarngpt_src")  # makes the cloned repo importable

        import torch
        from transformers import AutoModelForCausalLM
        from yarngpt.audiotokenizer import AudioTokenizerV2

        tokenizer_path = "saheedniyi/YarnGPT2"
        self.audio_tokenizer = AudioTokenizerV2(
            tokenizer_path,
            "/root/wavtokenizer_model.ckpt",
            "/root/wavtokenizer_config.yaml",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            tokenizer_path, torch_dtype="auto"
        ).to(self.audio_tokenizer.device)

    @modal.method()
    def synthesize(self, text: str, language: str, speaker_name: str) -> bytes:
        import io

        import torchaudio

        valid_voices = YARNGPT_VOICES.get(language.lower(), [])
        if speaker_name not in valid_voices:
            raise ValueError(
                f"'{speaker_name}' isn't a valid voice for language '{language}'. "
                f"Valid options: {valid_voices}"
            )

        prompt = self.audio_tokenizer.create_prompt(
            text, lang=language.lower(), speaker_name=speaker_name
        )
        input_ids = self.audio_tokenizer.tokenize_prompt(prompt)

        output = self.model.generate(
            input_ids=input_ids,
            temperature=0.1,
            repetition_penalty=1.1,
            max_length=4000,
        )
        codes = self.audio_tokenizer.get_codes(output)
        audio = self.audio_tokenizer.get_audio(codes)

        buffer = io.BytesIO()
        torchaudio.save(buffer, audio, sample_rate=24000, format="wav")
        return buffer.getvalue()

    @modal.fastapi_endpoint(method="POST")
    def synthesize_endpoint(self, request: YarnGPTRequest, authorization: str = None):
        import base64

        _check_auth(authorization)
        audio_bytes = self.synthesize.local(
            request.text, request.language, request.speaker_name
        )
        return YarnGPTResponse(audio_base64=base64.b64encode(audio_bytes).decode("utf-8"))
