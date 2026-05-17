"""
ReAct Agent using Gemma 4 native tool-calling protocol.
Tools: get_plant_name (vision), get_disease_name (text)
"""

import re
import json
import torch
from transformers import AutoProcessor, AutoModelForCausalLM  # ← fix #2
from PIL import Image

# ─── Config ───────────────────────────────────────────────────────────────────

# MODEL_ID = "google/gemma-4-e2b-it"
MODEL_ID = "google/gemma-4-e2b-it"
MAX_STEPS = 5
MAX_NEW_TOKENS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load Model ───────────────────────────────────────────────────────────────

print(f"Loading {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(  # ← fix #2
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map=DEVICE
)
model.eval()
print("Model loaded.")



def extract_tool_calls(text: str) -> list[dict]:
    def cast(v):
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return {"true": True, "false": False}.get(v.lower(), v.strip("'\""))

    return [
        {
            "name": name,
            "arguments": {
                k: cast((v1 or v2).strip())
                for k, v1, v2 in re.findall(
                    r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]*))', args
                )
            },
        }
        for name, args in re.findall(
            r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", text, re.DOTALL
        )
    ]



# ─── Generate helper ──────────────────────────────────────────────────────────

def build_initial_user_message(text: str, images: list | None) -> dict:
    """
    *** FIX #1 ***
    Build a properly structured multimodal user message.
    Content must be a list of typed blocks, NOT a string with <image> prepended.
    """
    if images:
        content = []
        for img in images:
            content.append({"type": "image", "image": img})  # ← correct format
        content.append({"type": "text", "text": text})
        return {"role": "user", "content": content}
    else:
        return {"role": "user", "content": text}


def generate(messages: list, tools: list | None = None, images: list | None = None) -> str:
    """Run one generation pass. Images only needed on the first call."""

    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools

    # apply_chat_template handles image token injection automatically
    # when content is a list with {"type": "image"} blocks
    text = processor.apply_chat_template(messages, **kwargs)

    # Debug: show if tool_response tokens are present in rendered text
    if "<tool_response" in text or "tool_response" in text:
        print("[DEBUG] ✓ tool_response tokens found in rendered prompt")
    else:
        print("[DEBUG] ✗ NO tool_response tokens in rendered prompt")
    print(f"[DEBUG] Prompt length: {len(text)} chars")

    # Collect PIL images from the message content to pass to processor
    # (processor needs them separately alongside the templated text)
    pil_images = images  # already extracted upstream; None on subsequent steps

    if pil_images:

        print("COmming pil_images")
        inputs = processor(
            text=text,
            images=pil_images,
            return_tensors="pt"
        ).to(model.device)
        print(f"[DEBUG] pixel_values shape: {inputs['pixel_values'].shape}")
    else:
        inputs = processor(text=text, return_tensors="pt").to(model.device)

    with torch.inference_mode() : 
        print("with torch.inference_mode::::::")
        # print("text:::::::;;\n\n" , text ,"\n")
        # print(f"[DEBUG] Input token count: {inputs['input_ids'].shape[-1]}")
        # print(text.count("<image_soft_token>"))  # or whatever Gemma's token string is
        # print(text.count("<image>"))

        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    generated_tokens = out[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(generated_tokens, skip_special_tokens=False)

# ─── ReAct Agent Loop ─────────────────────────────────────────────────────────

def react_agent(user_prompt: str, images: list | None = None , tools : list|None=None) -> str:
    messages = [
        {
            "role": "system",
            "content": """You are a Plant Disease Information Extraction Agent.
                    Your task is to analyze unstructured raw text about plant diseases and extract key information into a clean structured JSON format.

                    Instructions:
                    - Read the input carefully.
                    - Extract only information explicitly mentioned or strongly implied.
                    - Normalize spelling and terminology where appropriate.
                    - Return concise entries.
                    - Avoid duplicates.
                    - If a category is missing, return an empty array.
                    - Output ONLY valid JSON.
                    - Do not include explanations, markdown, or extra text.

                    Required Output Format:
                    {
                    "disease_name" : "identified disease",
                    "symptoms": [],
                    "hosts": [],
                    "pathogens": [],
                    "precautions": [],
                    "recommendations": []
                    }


                    Extraction Rules:
                    - symptoms: visible signs or plant conditions caused by the disease.
                    - hosts: affected crops, plants, or species.
                    - pathogens: fungi, bacteria, viruses, nematodes, or causal organisms.
                    - precautions: preventive measures to avoid spread or infection.
                    - recommendations: treatment, management, monitoring, or recovery actions.

                    Guidelines:
                    - Keep outputs short and standardized.
                    - Convert plural crop names into singular where appropriate.
                    - Remove duplicate or overlapping entries.
                    - Do not invent information not present in the input.
                    - If uncertain, omit the item instead of guessing.

                    Example Input:
                    Tomato early blight is caused by Alternaria solani. Symptoms include concentric brown leaf spots, yellowing leaves, and premature defoliation. The disease mainly affects tomato and potato plants. Farmers should avoid overhead irrigation and rotate crops regularly. Recommended control measures include applying fungicides and removing infected plant debris.

                    Example Output:
                    {
                    "disease_name" : "Armillaria",
                    "symptoms": [
                        "concentric brown leaf spots",
                        "yellowing leaves",
                        "premature defoliation"
                    ],
                    "hosts": [
                        "tomato",
                        "potato"
                    ],
                    "pathogens": [
                        "Alternaria solani"
                    ],
                    "precautions": [
                        "avoid overhead irrigation",
                        "rotate crops regularly"
                    ],
                    "recommendations": [
                        "apply fungicides",
                        "remove infected plant debris"
                    ]
                    }  """
        },
        # *** FIX #1 applied here ***
        build_initial_user_message(user_prompt, images),
    ]

    for step in range(MAX_STEPS):
        print(f"\n{'='*60}\nStep {step + 1}\n{'='*60}")

        # *** FIX #3: only pass raw PIL images on step 0 ***
        # They're already encoded into the message content structure above;
        # we just need them available for processor() to build pixel_values
        output = generate(messages, tools=tools, images=images)
        calls = extract_tool_calls(output)

        if not calls:
            clean = re.sub(r"<\|?\w+\|?>", "", output).strip()
            print(f"[FINAL ANSWER] {clean}")
            return clean

        print(f"[TOOL CALLS] {json.dumps(calls, indent=2)}")
        results = []
        for c in calls:
            res = execute_tool(c)
            results.append({"name": c["name"], "response": res})
            print(f"  → {c['name']}() = {json.dumps(res)}")

        print("tool response::", results)

        # Append as a SINGLE assistant message with tool_calls + tool_responses
        # This is the format apply_chat_template expects (per official Gemma 4 docs)
        assistant_msg = {
            "role": "assistant",
            "tool_calls": [{"function": c} for c in calls],
            "tool_responses": results,
        }
        messages.append(assistant_msg)

        # Debug: confirm messages list is growing
        print(f"[DEBUG] messages length after step {step + 1}: {len(messages)}")
        print(f"[DEBUG] last message appended: {json.dumps(assistant_msg, indent=2, default=str)}")

        # After first step, don't re-send images (already encoded in first turn)

    print(f"\n[MAX STEPS REACHED] Forcing final answer...")
    output = generate(messages, tools=None, images=images)
    clean = re.sub(r"<\|?\w+\|?>", "", output).strip()
    print(f"[FINAL ANSWER] {clean}")
    return clean

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    BATCH_SIZE = 5
    json_path = './dataset/diseases.json'

    with open(json_path, 'r') as f:
        json_file = json.load(f)

    def process_doc(doc):
        """Process a single document through the agent and return enriched data."""
        doc_id = doc.get("id")
        print("doc_id_processing::" , doc_id)
        user_query = json.dumps(doc, indent=2)

        try:
            answer = react_agent(user_query, images=[], tools=[])
            json_response = json.loads(answer)
            return {
                "id": doc_id,
                "symptoms": json_response.get("symptoms", []),
                "pathogens": json_response.get("pathogens", []),
                "precautions": json_response.get("precautions", []),
                "recommendations": json_response.get("recommendations", []),
                "hosts": json_response.get("hosts", []),
                "disease_name" :  json_response.get("disease_name", []),


            }
        except Exception as e:
            print(f"[ERROR] Failed on doc {doc_id}: {e}")
            return {"id": doc_id, "error": str(e)}

    # Process in batches of 5
    total = len(json_file)
    for batch_start in range(0, total, BATCH_SIZE):
        batch = json_file[batch_start:batch_start + BATCH_SIZE]
        print(f"\n{'='*60}")
        print(f"Processing batch {batch_start // BATCH_SIZE + 1} "
              f"(docs {batch_start} to {batch_start + len(batch) - 1})")
        print(f"{'='*60}")

        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = {executor.submit(process_doc, doc): doc for doc in batch}

            for future in as_completed(futures):
                result = future.result()
                if "error" in result:
                    continue

                # Update the original json_file entry
                for ele in json_file:
                    if ele.get("id") == result["id"]:
                        ele["symptoms"] = result["symptoms"]
                        ele["pathogens"] = result["pathogens"]
                        ele["precautions"] = result["precautions"]
                        ele["recommendations"] = result["recommendations"]
                        ele["hosts"] = result["hosts"]
                        ele["disease_name"] = result["disease_name"]
                        break

        # Save after each batch (so progress isn't lost)
        with open(json_path, 'w') as f:
            json.dump(json_file, f, indent=4)
        print(f"[SAVED] Batch complete, {json_path} updated.")
        break

    print(f"\nDone. Processed {total} documents.")
