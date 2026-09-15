# Setup & Run Guide — Grounded RAG PDF/Word Summarizer

This covers everything from a completely fresh machine to a working local
test. Deployment to Hugging Face Spaces is included at the end since that's
where the final app will live, but you can stop after "Local testing" if you
just want to confirm things work first.

---

## 0. Prerequisites

- Python 3.10+ installed
- A Hugging Face account (free) — for the gated Llama model and hosting the Space
- A Modal account (free to sign up, pay-as-you-go for GPU usage) — https://modal.com
- An ElevenLabs account — https://elevenlabs.io (free tier has a limited monthly character quota; fine for testing)

---

## 1. Get repo access to the gated Llama model

`meta-llama/Llama-3.1-8B-Instruct` is gated — you need to request access once:

1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Click "Agree and access repository" (approval is usually near-instant)
3. Go to https://huggingface.co/settings/tokens → create a token with **read** access
4. Keep this token handy — you'll use it as `HF_TOKEN` in step 3

---

## 2. Install and authenticate Modal

```bash
pip install modal
modal setup
```

This opens a browser window to link your Modal account to the CLI. Once linked, you're ready to create secrets.

---

## 3. Create Modal secrets

Modal secrets are how `modal_app.py` gets credentials without them being hardcoded anywhere:

```bash
modal secret create hf-token HF_TOKEN=hf_your_actual_token_here

modal secret create rag-auth-token RAG_AUTH_TOKEN=choose_a_long_random_string_here
```

- `hf-token` → your Hugging Face token from step 1 (needed to download the gated Llama model)
- `rag-auth-token` → any long random string you make up yourself. This is the shared secret your Streamlit app will send to prove it's allowed to call your Modal endpoints. Generate one quickly with:
  ```bash
  python3 -c "import secrets; print(secrets.token_urlsafe(32))"
  ```

---

## 4. Deploy the Modal app

From the project directory (wherever `modal_app.py` lives):

```bash
modal deploy modal_app.py
```

First deploy will take a while — it's building two container images (one with PyTorch/transformers/bitsandbytes, one with YarnGPT's repo + WavTokenizer weights, which involves cloning a GitHub repo and downloading model weights during the build).

When it finishes, Modal prints out **four HTTPS URLs**, one per endpoint:

```
├── 🔨 Created embed_endpoint => https://your-workspace--pdf-rag-summarizer-embedder-embed-endpoint.modal.run
├── 🔨 Created summarize_endpoint => https://...-summarizer-summarize-endpoint.modal.run
├── 🔨 Created translate_endpoint => https://...-summarizer-translate-endpoint.modal.run
└── 🔨 Created synthesize_endpoint => https://...-yarngptsynthesizer-synthesize-endpoint.modal.run
```

**Copy all four** — you'll need them as environment variables next.

---

## 5. Test an endpoint before wiring up Streamlit

Sanity-check the embedding endpoint works end to end:

```bash
curl -X POST "https://your-embed-endpoint-url" \
  -H "Authorization: Bearer choose_a_long_random_string_here" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello world"]}'
```

A JSON response with a list of numbers means it works. A `401` means the auth token doesn't match what you set in step 3. A timeout on first call is normal — that's the cold start (container spinning up); it'll be fast on the next call while the container stays warm.

---

## 6. Get your ElevenLabs API key

1. Log into https://elevenlabs.io → click your profile icon → "API Keys"
2. Create a key, copy it
3. You'll set this as `ELEVENLABS_API_KEY` in the next step

---

## 7. Set environment variables for local testing

Create a `.env` file (or export directly in your shell) in the project directory:

```bash
export MODAL_EMBED_URL="https://your-embed-endpoint-url"
export MODAL_SUMMARIZE_URL="https://your-summarize-endpoint-url"
export MODAL_TRANSLATE_URL="https://your-translate-endpoint-url"
export MODAL_YARNGPT_URL="https://your-synthesize-endpoint-url"
export RAG_AUTH_TOKEN="choose_a_long_random_string_here"     # same value as the Modal secret
export ELEVENLABS_API_KEY="your_elevenlabs_key_here"
```

Run `source .env` (if using a file) before starting Streamlit, or export each line directly in your terminal session.

---

## 8. Install local dependencies and run

```bash
pip install -r requirements.txt
streamlit run main.py
```

**Note:** `main.py` hasn't been rewired to the new RAG pipeline yet in this build sequence — once that's done, this is the command that runs the actual app locally at `http://localhost:8501`.

---

## 9. Deploying to Hugging Face Spaces (when ready)

1. Go to https://huggingface.co/new-space → choose the **Streamlit** SDK, free CPU tier
2. Push your code (excluding `modal_app.py` — that stays deployed on Modal, not the Space):
   ```bash
   git remote add space https://huggingface.co/spaces/your-username/your-space-name
   git push space main
   ```
3. In the Space's **Settings → Variables and secrets**, add the same six values from step 7 as **secrets** (not public variables, since they include credentials):
   - `MODAL_EMBED_URL`, `MODAL_SUMMARIZE_URL`, `MODAL_TRANSLATE_URL`, `MODAL_YARNGPT_URL`, `RAG_AUTH_TOKEN`, `ELEVENLABS_API_KEY`
4. The Space rebuilds automatically and should come up at `https://huggingface.co/spaces/your-username/your-space-name`

---

## Troubleshooting quick reference

| Symptom | Likely cause |
|---|---|
| `401 Unauthorized` from any Modal endpoint | `RAG_AUTH_TOKEN` doesn't match between your Modal secret and your app's env var |
| Very slow first request after idle | Normal cold start — container was scaled to zero and is spinning back up |
| `modal deploy` fails during YarnGPT image build | The `gdown` download step can occasionally get rate-limited by Google Drive on repeated builds; wait a bit and retry |
| Llama model fails to load with a 403/gated error | You haven't been granted access yet on the Hugging Face model page, or `HF_TOKEN` in the Modal secret is wrong/expired |
| ElevenLabs calls fail with quota error | Free tier monthly character limit reached. Note: currently there's no automatic fallback for this — ElevenLabs handles English and YarnGPT handles Yoruba/Igbo/Hausa, but if ElevenLabs itself fails there's no backup engine for English yet (this was flagged as an open decision earlier in the build) |
