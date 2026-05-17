"""
ReAct Agent using Ollama with native tool-calling.
Tools: get_disease_name (vector search), get_disease_recommendation (JSON lookup)
"""

import re
import json
import time
import base64
import io
import ollama
from PIL import Image
from embedding_search import get_nearest_docs

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_ID = "gemma4:e2b"
MAX_STEPS = 5
JSON_PATH = "./dataset/diseases.json"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    DISEASES_DB = json.load(f)

# ─── Tool Implementations ─────────────────────────────────────────────────────


def get_disease_name(details_of_symptoms: str , plant_name : str) -> dict:
    """Search vector DB for diseases matching the described symptoms."""

    return get_nearest_docs(details_of_symptoms , plant_name , 2)


def get_disease_recommendation(disease_name: str, host_name: str) -> list[dict]:
    """
    Search diseases.json for entries matching both:
      - host_name (case-insensitive) in the 'hosts' list
      - disease_name (regex, case-insensitive) matching 'disease_name' field
    """
    results = []
    pattern = re.compile(disease_name, re.IGNORECASE)

    for doc in DISEASES_DB:
        hosts = doc.get("hosts", [])
        host_match = any(host_name.lower() in h.lower() for h in hosts)
        doc_disease = doc.get("disease_name", "")
        disease_match = bool(pattern.search(doc_disease))

        if host_match and disease_match:
            results.append({
                "disease_name": doc_disease,
                "hosts": hosts,
                "precautions": doc.get("precautions", []),
                "recommendations": doc.get("recommendations", []),
                "citation": doc.get("citation", []),
            })
    if not results:
        for doc in DISEASES_DB:
            doc_disease = doc.get("disease_name", "")
            disease_match = bool(pattern.search(doc_disease))
            if disease_match:
                results.append({
                    "disease_name": doc_disease,
                    "hosts": doc.get("hosts", []),
                    "precautions": doc.get("precautions", []),
                    "recommendations": doc.get("recommendations", []),
                    "citation": doc.get("citation", []),
                })

    return results


# ─── Tool Registry & Schemas ──────────────────────────────────────────────────

TOOL_REGISTRY = {
    "get_disease_recommendation": get_disease_recommendation,
    "get_disease_name": get_disease_name,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_disease_recommendation",
            "description": (
                "Use this tool after the disease name and host plant have been identified. "
                "Returns treatment advice, precautions, recommendations, and citations. "
                "ALL ARGUMENTS MUST BE IN ENGLISH ONLY — translate from the user's language before calling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "disease_name": {
                        "type": "string",
                        "description": "The disease name in ENGLISH ONLY (e.g., 'early blight', 'powdery mildew'). Translate from the user's language if needed. Never pass non-English text.",
                    },
                    "host_name": {
                        "type": "string",
                        "description": "The host or plant name in ENGLISH ONLY (e.g., 'tomato', 'rice', 'wheat'). Translate from the user's language if needed. Never pass non-English text.",
                    },
                },
                "required": ["disease_name", "host_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_disease_name",
            "description": (
                "Given a description of symptoms observed on a plant (in ENGLISH), returns the most "
                "likely disease name and its full symptom profile. "
                "ALL ARGUMENTS MUST BE IN ENGLISH ONLY — translate from the user's language before calling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "details_of_symptoms": {
                        "type": "string",
                        "description": "A text description of the symptoms observed on the plant (from user query and/or image) in ENGLISH ONLY. Translate from the user's language if needed. Never pass non-English text.",
                    },
                    "plant_name": {
                        "type": "string",
                        "description": "The name of the plant confirmed by the user, in ENGLISH ONLY (e.g., 'tomato', 'rice'). Translate from the user's language if needed. Never pass non-English text.",
                    },

                    
                },
                "required": ["details_of_symptoms" , "plant_name"],
            },
        },
    },
]

# ─── Helpers ──────────────────────────────────────────────────────────────────


def pil_to_base64(img: Image.Image) -> str:
    """Convert a PIL image to base64 string for Ollama."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def execute_tool(name: str, args: dict) -> dict:
    """Dispatch a tool call to the actual function."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        result = fn(**args)
        return result if isinstance(result, dict) else {"result": result}
    except Exception as e:
        return {"error": str(e)}


# ─── Generate (Ollama) ────────────────────────────────────────────────────────


def generate(messages: list, tools: list | None = None):
    """
    Call Ollama chat API. Returns the response message object.
    Images should already be embedded in messages as base64.
    """

    try:
        kwargs = {
            "model": MODEL_ID,
            "messages": messages,
            "options": {
                "num_predict": 100000,
                # "keep_alive" : -1
                },
            "think": "medium",
            "keep_alive": -1
        }

        if tools:
            kwargs["tools"] = tools

        st = time.time()
        response = ollama.chat(**kwargs)
        elapsed = time.time() - st

        # Handle both object-style (ollama >= 0.4) and dict-style (older) responses
        if hasattr(response, "eval_count"):
            gen_tokens = response.eval_count or 0
            gen_time = (response.eval_duration or 0) / 1e9
            prompt_tokens = response.prompt_eval_count or 0
            msg = response.message
        else:
            gen_tokens = response.get("eval_count", 0)
            gen_time = response.get("eval_duration", 0) / 1e9
            prompt_tokens = response.get("prompt_eval_count", 0)
            msg = response.get("message", {})

        tps = gen_tokens / gen_time if gen_time > 0 else 0
        print(f"[DEBUG] Prompt: {prompt_tokens} tok | Generated: {gen_tokens} tok in {gen_time:.2f}s → {tps:.1f} tok/s | Total: {elapsed:.2f}s")

        # Normalize to a plain dict so react_agent can use .get() consistently
        if hasattr(msg, "role"):
            tool_calls = None
            if msg.tool_calls:
                tool_calls = [
                    {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            return {
                "role": msg.role,
                "content": msg.content or "",
                "tool_calls": tool_calls,
            }

        return msg
    except Exception as e:
        print("Exception in generate as" , str(e))
        raise e


# ─── ReAct Agent Loop ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Groot, an expert Plant Diagnosis Assistant specialized in identifying plant and leaf diseases using:
- User-described symptoms
- Uploaded plant/leaf images (optional)
- Diagnostic tools (get_disease_name, get_disease_recommendation)

Your primary task is to provide the most accurate possible plant disease diagnosis based on confirmed evidence only.

================================================================
## TOP-PRIORITY RULE — TWO SEPARATE LANGUAGES (READ FIRST)
================================================================
There are TWO completely independent languages you MUST manage. Do not confuse them.

1. **TOOL PARAMETER LANGUAGE → ALWAYS ENGLISH. NO EXCEPTIONS.**
   - Every argument value passed to ANY tool (`get_disease_name`, `get_disease_recommendation`)
     MUST be written in **English only** — even if the user wrote in Hindi, Spanish,
     French, Tamil, Marathi, Bengali, Arabic, or any other language.
   - This applies to: `plant_name`, `host_name`, `disease_name`, `details_of_symptoms`.
   - Before calling a tool, you MUST mentally translate the user's content into English
     and pass the English translation as the argument.
   - The vector database and JSON dataset are indexed in English. Passing any other
     language will cause the tool to fail or return wrong results.

   CORRECT examples (regardless of user's language):
      get_disease_name(plant_name="tomato", details_of_symptoms="yellow spots on leaves with brown edges")
      get_disease_recommendation(disease_name="early blight", host_name="tomato")

   WRONG — NEVER do this:
      get_disease_name(plant_name="टमाटर", details_of_symptoms="पत्तियों पर पीले धब्बे")
      get_disease_name(plant_name="tomate", details_of_symptoms="manchas amarillas en las hojas")
      get_disease_recommendation(disease_name="झुलसा रोग", host_name="टमाटर")

2. **USER RESPONSE LANGUAGE → `{language}`**
   - Your visible reply to the user (text the user reads) must be entirely in `{language}`.
   - This is independent from rule #1. Tool args stay English even when the reply is in `{language}`.

**Mental checklist before EVERY tool call:**
   [ ] Are all argument values in English? If no → translate to English first.
   [ ] Did the user write in another language? Doesn't matter → still English for tool args.
   [ ] Am I responding to the user? → that text goes in `{language}`, not the tool args.

================================================================

## TOOLS AVAILABLE:

### 1. `get_disease_name`
- **When to use:** Once BOTH confirmed inputs are available.
- **Inputs (ENGLISH ONLY):**
  - `plant_name` — confirmed plant name, translated to English
  - `details_of_symptoms` — confirmed symptom description, translated to English
- **Returns:** Most likely disease name + full symptom profile + confidence scores.

### 2. `get_disease_recommendation`
- **When to use:** After the disease name has been confirmed by the user.
- **Inputs (ENGLISH ONLY):**
  - `disease_name` — confirmed disease name, in English
  - `host_name` — confirmed plant name, in English
- **Returns:** Treatment advice, precautions, care recommendations, and citations.

## WORKFLOW — FOLLOW THIS STRICTLY IN ORDER:

### STEP 1: INFORMATION CONFIRMATION GATE
Before calling any tool, you MUST confirm the following two inputs:

| Required Input       | Source                              | Status Check                          |
|----------------------|-------------------------------------|----------------------------------------|
| `plant_name`         | Explicitly stated or confirmed by user (if not provided) | Never assume from image alone        |
| `details_of_symptoms`| Described by user AND/OR observed in image | More details of the symptoms along with what you see for a complete information |

**Rules:**
- If an image is uploaded: extract visible symptoms from it and confirm from the user what you see (plant and symptoms) and ask the user to confirm it (if user has not stated it).
- If either input is missing or unclear, ask a targeted follow-up question to obtain it.
- Do NOT proceed to Step 2 until BOTH inputs are confirmed.
- Do NOT assume plant name from image alone — always get user confirmation.

**Ask follow-up questions in `{language}`. Internally store the confirmed answer translated to English for tool use.**

### STEP 2: CALL `get_disease_name`
- Once BOTH `plant_name` and `details_of_symptoms` are confirmed, call `get_disease_name`.
- **Translate both arguments to English before passing them.** This is mandatory.
- Pass the full confirmed symptom details — never partial information.

### STEP 3: PROCESS TOOL RESPONSE

#### If one disease is returned with high confidence:
- Present the disease name, matched symptoms, and confidence score (in `{language}`).
- Ask the user to confirm the diagnosis before proceeding.

#### If multiple diseases are returned:
- Rank them by confidence score (from tool output).
- Present as a numbered list with confidence percentages.
- Ask concise clarifying questions to narrow down (guided by the symptom differences returned by the tool).

#### If the tool returns no result or low confidence:
- Inform the user clearly that you could not confidently identify the disease.
- Ask for additional details such as:
  - Duration of symptoms
  - Environmental conditions (humidity, temperature, recent weather)
  - Any treatments already attempted
  - A clearer or closer image of the affected area
- Then retry `get_disease_name` with the updated information (still in English).

### STEP 4: USER CONFIRMS DISEASE
- Wait for explicit user confirmation of the diagnosed disease.
- Do NOT call `get_disease_recommendation` before user confirms.

### STEP 5: CALL `get_disease_recommendation`
- Once the user confirms the disease, call `get_disease_recommendation` with:
  - `disease_name` (confirmed, in **English**)
  - `host_name` (confirmed plant name, in **English**)

### STEP 6: DELIVER FINAL RESPONSE
Using the tool response, translate the content into `{language}` and provide:
- **Disease Overview** — brief explanation of the disease
- **Treatment / Management** — actionable steps
- **Precautions** — how to prevent spread or recurrence
- **Citations** — from tool response if available

## USER RESPONSE LANGUAGE HANDLING (`{language}`):
- Every word the user reads must be in `{language}` — headings, bullets, labels, everything.
- Do NOT detect, guess, or switch language based on the user's message content.
- If a disease name or technical term has no good translation, keep it in English inside the `{language}` reply.
- Translate tool responses fully into `{language}` before presenting to the user.
- This rule applies ONLY to user-facing text, never to tool arguments.

## RESPONSE STYLE:
- Always respond in **Markdown** with clear headings and bullet points.
- Keep responses brief, structured, and technically accurate.
- Prefer short sections over long paragraphs.
- Focus only on actionable and relevant information.
- Do not share tool information or talk about tool calling with the user.

## STRICT CONSTRAINTS — NEVER VIOLATE:
- NEVER pass non-English text to any tool. Translate to English first, every single time.
- Do NOT call any tool with unconfirmed or assumed data.
- Do NOT diagnose any disease without tool output.
- Do NOT assume plant name — always confirm with the user.
- Do NOT call `get_disease_recommendation` before the user confirms the disease name.
- Do NOT fabricate diseases, symptoms, confidence scores, or treatments.
- Do NOT skip the Information Confirmation Gate (Step 1).
- Always communicate uncertainty clearly when confidence is low.
- Always base confidence percentages on actual tool output scores.
"""




MESSAGES = [ ]


def react_agent(user_prompt: str, images: list | None = None, tools: list | None = None, language: str = "English") -> str:
    """ReAct loop using global MESSAGES for session memory."""

    print("language:::" , language)

    try:

        system_prompt_message = {"role": "system", "content": SYSTEM_PROMPT.format(language=language)}
        if not MESSAGES:
            MESSAGES.append(system_prompt_message)

        # Build user message
        if images:
            # Ollama expects images as base64 in the message
            img_data = [pil_to_base64(img) for img in images]
            MESSAGES.append({"role": "user", "content": user_prompt, "images": img_data})
        else:
            MESSAGES.append({"role": "user", "content": user_prompt})

        for step in range(MAX_STEPS):
            print(f"\n{'='*60}\nStep {step + 1}\n{'='*60}")

            response_msg = generate(MESSAGES, tools=tools)

            # Check if model made tool calls
            tool_calls = response_msg.get("tool_calls")

            if not tool_calls:
                # No tool calls → final text answer
                answer = response_msg.get("content", "").strip()
                print(f"[FINAL ANSWER] {answer[:200]}...")
                MESSAGES.append({"role": "assistant", "content": answer})
                return answer

            # Process tool calls
            print(f"[TOOL CALLS] {json.dumps(tool_calls, indent=2, default=str)}")

            # Append assistant message with tool calls only (strip None fields)
            assistant_msg = {"role": "assistant", "content": response_msg.get("content", "")}
            assistant_msg["tool_calls"] = tool_calls
            MESSAGES.append(assistant_msg)

            # Execute each tool and append results
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"]["arguments"]

                print(f"  → {fn_name}({fn_args})")
                result = execute_tool(fn_name, fn_args)
                print(f"    = {json.dumps(result, default=str)}")

                # Ollama expects tool results as role="tool" messages
                MESSAGES.append({
                    "role": "tool",
                    "content": json.dumps(result, default=str),
                })

        # Hit max steps — force final answer without tools
        print(f"\n[MAX STEPS REACHED] Forcing final answer...")
        response_msg = generate(MESSAGES, tools=None)
        answer = response_msg.get("content", "").strip()
        print(f"[FINAL ANSWER] {answer[:200]}...")
        MESSAGES.append({"role": "assistant", "content": answer})
        return answer

    except Exception as e:
        print("Exception in react_agent as" , str(e))
        raise e