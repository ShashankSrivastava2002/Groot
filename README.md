# 🌱 Groot — Plant Disease AI Assistant

A multilingual, multimodal plant disease diagnosis assistant powered by **Gemma 4**.
Runs entirely on local hardware. Accepts text, voice, or image queries in **7 languages**.
Every diagnosis is grounded in real agricultural research with cited sources.

---

## Table of Contents

- [Features](#-features)
- [Setup](#-setup)
  - [Option A: Docker (recommended)](#option-a-docker-recommended)
  - [Option B: Local development](#option-b-local-development)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Logical Flow](#-logical-flow)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Tech Stack](#-tech-stack)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖼️ **Image input** | Upload a photo of a sick leaf — Gemma 4's native vision encoder analyzes it |
| 🎤 **Voice input** | Record symptoms in any language — Whisper-small transcribes locally |
| 🌐 **7 languages** | English, French, Spanish, German, Chinese, Japanese, Hindi |
| 🔍 **Grounded diagnosis** | Every answer comes from a vector-searched agricultural knowledge base |
| 📚 **Cited sources** | Recommendations link directly to UC IPM research pages |
| 💬 **Multi-turn ReAct loop** | Asks clarifying questions, doesn't guess from one photo |
| 🔒 **Fully local** | No API keys, no cloud calls, your data never leaves your machine |
| 🐳 **One-command deploy** | `docker compose up -d` and you're live |

---

## 🚀 Setup


### Option A: Local development

( NOT USING DOCKER AS: TRANSFORMERS , TORCH , OLLAMA , GEMMA-E2B will combinly make a image of 20 GB , hard to load for most of people)

For active development on the Python code or UI.

**Prerequisites:**
- Python 3.11+
- ~10 GB disk space for the Gemma 4 model
- A working microphone (optional, for voice)

#### 1. Install Ollama and pull Gemma 4

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma4:e2b
```

Verify Ollama is running on `http://localhost:11434`:
```bash
ollama list
```

#### 2. Set up the Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-docker.txt
```

#### 3. (Optional) Rebuild the vector database

The `database/` folder ships with a pre-built ChromaDB collection. To rebuild it from scratch:

```bash
# Step 1: Scrape disease records (one-time, ~5 min)
python3 scrap.py

# Step 2: Enrich with structured fields (uses Gemini API, set GOOGLE_API_KEY first)
export GOOGLE_API_KEY="your-key-here"
python3 enrichment_data.py

# Step 3: Build vector embeddings into ChromaDB
python3 tests/db_vector.py
```

#### 4. Run the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000**.

---

## 🎯 Usage

1. Open the app in your browser.
2. **Pick your language** from the dropdown (top right).
3. **Describe your problem** — type, speak (🎤), or upload a photo (📷).
4. Groot asks clarifying questions. Answer them.
5. Once it has a confident match, it presents the diagnosis with confidence percentages.
6. **Confirm the diagnosis** when prompted.
7. Groot returns treatment recommendations, precautions, and a citation link.

> **Tip:** For best results, include the **plant species** ("my tomato plant") in your first message and describe symptoms in detail.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│  Browser UI (HTML / Vanilla JS)                        │
│   • Markdown rendering (marked + DOMPurify)            │
│   • Image upload + preview                             │
│   • MediaRecorder → WAV → /transcribe                  │
│   • Language dropdown                                  │
└─────────────────────┬──────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│  FastAPI Server                                        │
│   /chat, /transcribe, /reset, /                        │
└─────────┬──────────────────────────┬───────────────────┘
          │                          │
          ▼                          ▼
   ┌──────────────┐           ┌─────────────────┐
   │ Whisper-small│           │  ReAct Agent    │
   │  (transcribe)│           │  (loop manager) │
   └──────────────┘           └────────┬────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
      ┌──────────────┐        ┌──────────────────┐    ┌─────────────────┐
      │   Ollama +   │        │   ChromaDB +     │    │  Disease JSON   │
      │   Gemma 4    │        │   E5 embeddings  │    │  (regex lookup) │
      └──────────────┘        └──────────────────┘    └─────────────────┘
```

---

## 🔄 Logical Flow

This section explains how a single user query travels through the system end-to-end.

### Section 1: Knowledge Base Construction (one-time, offline)

Before users ever touch the app, we build the dataset:

1. **Scrape** disease records from [UC IPM](https://ipm.ucanr.edu/) using BeautifulSoup. Each entry has a host plant, symptom description, scientific name, and disease type.
2. **Enrich** raw HTML into structured JSON using the Gemini API. We extract: `symptoms[]`, `hosts[]`, `pathogens[]`, `precautions[]`, `recommendations[]`, and `citation[]` (URLs).
3. **Embed** each disease's symptom list using `intfloat/e5-base-v2` (768-dim).
4. **Index** into ChromaDB with disease name, hosts, and pathogens as metadata.

The result: ~200 plant diseases stored as searchable vector passages.

---

### Section 2: Gemma 4 as the Reasoning Engine

We use **Gemma 4 (gemma4:e2b)** for three reasons:

- **Native multimodality** — image and text in the same turn, no separate vision model.
- **Native tool calling** — emits structured tool-call tokens when given function schemas.
- **Resource efficiency** — runs comfortably on a 16 GB laptop.

Gemma 4 is served through **Ollama** for persistent model loading and an OpenAI-compatible chat API.

---

### Section 3: The ReAct Agent Loop

The core of Groot is a **reason-and-act loop** in `tool_call_gemma.py`:

```
loop until final answer or MAX_STEPS=5:
    1. Call Gemma 4 with [system prompt, conversation history, tools]
    2. If response has tool_calls:
         → execute each tool
         → append result as { role: "tool", content: ... }
         → loop again
       Else:
         → return the text answer
```

The system prompt enforces an **Information Confirmation Gate**:

> "Before calling any tool, you MUST confirm BOTH `plant_name` AND `details_of_symptoms`. If either is missing, ask a follow-up question."

This single rule eliminated the most common failure — hallucinating diagnoses from one ambiguous photo.

---

### Section 4: The Tools

The agent has access to two tools defined in `tool_call_gemma.py`:

#### `get_disease_name(details_of_symptoms, plant_name)`
Vector search for top-N matching diseases.

```python
# In embedding_search.py
def get_nearest_docs(query_text, plant_name, n_results=10):
    embedding = e5_model(query_text)
    results = chromadb.query(embedding, n_results)
    # Filter: confidence ≥ 80% AND host plant matches
    return filtered_results
```

Returns disease name, matched symptoms, hosts, pathogens, and a confidence score.

#### `get_disease_recommendation(disease_name, host_name)`
Lookup over the structured JSON dataset.

```python
# In tool_call_gemma.py
def get_disease_recommendation(disease_name, host_name):
    # Regex match disease + substring match host
    return precautions, recommendations, citations
```

Returns precautions, treatment recommendations, and source URLs.

**Why two tools?** One is for **finding** the disease, the other is for **acting** on it. The agent never combines them prematurely — `get_disease_recommendation` is only called after the user confirms the diagnosis.

---

### Section 5: Multimodality

Three input modalities, all handled by the same `/chat` endpoint:

| Input | Browser side | Server side |
|---|---|---|
| **Text** | Plain `<textarea>` | Forwarded as-is to the agent |
| **Image** | `<input type="file">` → uploaded as `multipart/form-data` | Decoded with PIL, base64-encoded, embedded in the user message — Gemma 4 reads it natively |
| **Voice** | `MediaRecorder` → 16 kHz mono PCM WAV | `/transcribe` endpoint runs Whisper-small, returns text into the input box |

The voice path is two steps so the user can review and edit the transcription before sending — keeps things accurate.

---

### Section 6: Multilinguality

Groot supports 7 languages with two key design choices:

- **Sticky language per session.** The chosen language is injected into the system prompt at the first turn. Switching mid-conversation triggers a confirm dialog and a full reset. This eliminates language-drift hallucinations.
- **English tool parameters always.** Tool inputs (`disease_name`, `plant_name`, etc.) are always passed in English regardless of the user's language. This keeps vector queries consistent against a single English knowledge base. Gemma 4 translates the final response back into the user's language.

---

### Section 7: End-to-End Example

A user opens the app in **Hindi** mode and uploads a tomato leaf photo with brown spots.

```
1. UI: User picks "Hindi" from dropdown → selectedLanguage = "Hindi"
2. UI: User uploads image and types "मेरे टमाटर के पौधे पर भूरे धब्बे हैं"
3. UI: POSTs /chat with FormData { message, image, language }
4. FastAPI: Decodes image → calls react_agent(prompt, [image], tools, "Hindi")
5. Agent: Sets system prompt → Hindi locked
6. Agent: Sends [system, user(image+text)] to Gemma 4
7. Gemma 4: Sees both inputs, but hasn't confirmed plant_name explicitly
   → Asks (in Hindi): "क्या यह आपका टमाटर का पौधा है? मुझे और लक्षण बताएं।"
8. User: "Yes, tomato. The spots have rings inside, like targets."
9. Gemma 4: Now has plant_name + symptoms confirmed
   → Calls get_disease_name("brown spots with concentric rings", "tomato")
10. Tool: Vector search returns "Early Blight" with 87% confidence
11. Gemma 4: Presents in Hindi:
    "यह संभवतः **Early Blight** है (87% विश्वास)। क्या यह सही लगता है?"
12. User: "Yes, that matches."
13. Gemma 4: Calls get_disease_recommendation("Early Blight", "tomato")
14. Tool: Returns precautions + recommendations + citation URL
15. Gemma 4: Translates to Hindi, formats as markdown, returns final answer
16. UI: Renders markdown, makes URLs clickable, shows in chat
```

---

## 📁 Project Structure

```
Groot/
├── app.py                      # FastAPI server (entry point)
├── tool_call_gemma.py          # ReAct agent loop + tool definitions
├── embedding_search.py         # ChromaDB + E5 embeddings
├── transcription.py            # Whisper-small voice transcription
├── scrap.py                    # One-time UC IPM scraper
├── enrichment_data.py          # Gemini-based field extraction
├── templates/
│   └── index.html              # Single-page chat UI
├── dataset/
│   ├── diseases.json           # Raw scraped data
│   └── diseases_og2.json       # Enriched structured data
├── database/                   # ChromaDB persistent store
├── requirements-docker.txt     # Runtime dependencies
├── Dockerfile                  # App container build
├── docker-compose.yml          # Multi-container orchestration
├── .dockerignore
└── README.md
```

---

## 📡 API Reference

### `POST /chat`
Send a message + optional image to the agent.

**Form data:**
- `message` (str) — user query
- `language` (str, default `"English"`) — selected language
- `image` (file, optional) — uploaded image

**Response:** `{ "response": "<markdown text>" }`

### `POST /transcribe`
Transcribe an uploaded WAV file to text.

**Form data:**
- `audio` (file) — WAV blob (16 kHz mono PCM preferred)

**Response:** `{ "text": "<transcribed text>" }`

### `POST /reset`
Clear the server-side conversation history.

**Response:** `{ "status": "ok", "message": "Conversation reset." }`

### `GET /`
Serves the chat UI.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | **Gemma 4** (`gemma4:e2b`) via Ollama |
| LLM serving | Ollama (HTTP API on port 11434) |
| Web framework | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS, marked.js, DOMPurify |
| Vector DB | ChromaDB (HNSW, cosine similarity) |
| Embeddings | `intfloat/e5-base-v2` (768-dim) |
| Speech-to-text | Whisper-small (OpenAI, via HuggingFace) |
| Data enrichment (one-time) | Gemini 2.0 Flash |
| Containerization | Docker + Docker Compose |

---

## 🐛 Troubleshooting

**Ollama can't be reached:**
- Make sure Ollama is running: `ollama list`
- If using Docker, check the `OLLAMA_HOST` env var points to the right place
- Default port is 11434

**Microphone doesn't work:**
- Browser requires HTTPS for `getUserMedia` in production. Localhost works fine.
- Check browser permissions

**ChromaDB errors:**
- The `database/` folder must exist and be writable
- If corrupted, delete it and rebuild via `python3 tests/db_vector.py`

**Out of memory:**
- The Gemma 4 e2b model needs ~6 GB RAM during inference
- Try `gemma4:1b` for lower-spec devices: `ollama pull gemma4:1b` and update `MODEL_ID` in `tool_call_gemma.py`

---

## 📜 License

Open source. Use it. Improve it. Help farmers.

---

**Built with ❤️ and Gemma 4.**
