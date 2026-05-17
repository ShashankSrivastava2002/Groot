import torch.nn.functional as F
import chromadb, json, uuid, torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel



# ─── ChromaDB Setup ───────────────────────────────────────────────────────────

client = chromadb.PersistentClient(path="./database")

try:
    COLLECTION = client.create_collection(
        name="my_collection",
        metadata={"hnsw:space": "cosine"}
    )
except Exception:
    COLLECTION = client.get_collection(name="my_collection")


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    final_answer =  last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    embeddings = final_answer.cpu().numpy().tolist()
    return embeddings
    

TOKENIZER = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
EMBEDDING_MODEL = AutoModel.from_pretrained('intfloat/e5-base-v2')




def get_embeddings(documents: list):

        batch_dict = TOKENIZER(documents, max_length=5000, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = EMBEDDING_MODEL(**batch_dict)
            doc_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            return doc_embeddings


def format_results(res: dict, plant_name: str, n_results ,min_confidence: float = 80.0) -> list[dict]:
    """Convert raw ChromaDB query result into a clean list of dicts.
    Filters by host match and minimum confidence (default 80%)."""
    results = []
    ids = res["ids"][0]
    documents = res["documents"][0]
    metadatas = res["metadatas"][0]
    distances = res["distances"][0]

    for i in range(len(ids)):
        confidence = (1 - distances[i]) * 100

        # Skip low-confidence matches
        if confidence < min_confidence:
            continue

        host_present = plant_name.lower() in metadatas[i].get("hosts", " ").lower()

        # print("host_present::::", host_present, metadatas[i].get("hosts", " ").lower(), f"conf={confidence:.1f}%")

        if host_present:
            results.append({
                "document": documents[i],
                "disease_name": metadatas[i].get("disease_name", ""),
                "hosts": metadatas[i].get("hosts", " "),
                "pathogens": metadatas[i].get("pathogens", ""),
                # "confidence_score": confidence,
            })

    if results:
        results = results[: int(n_results)]
        print("LENTH OF TOOL format_results" , len(results))

    return results


def get_nearest_docs(query_text_list: list, plant_name: str, n_results: int = 10) -> list[dict]:
    """Search collection and return formatted results, filtered by plant_name in hosts."""
    doc_embeddings = get_embeddings(query_text_list)

    res = COLLECTION.query(
        query_embeddings=doc_embeddings,
        n_results=100
    )

    return format_results(res, plant_name , n_results)


# # Each input text should start with "query: " or "passage: ".
# # For tasks other than retrieval, you can simply use the "query: " prefix.

# query = 'tomato leaves have dark brown circles with rings inside them like a target'
# input_texts = [query ]
# # Tokenize the input texts
# results = get_nearest_docs(input_texts ,"ToMato", 2)

# print(results)

# for res in results:
#     print("="*100)
#     print(res)