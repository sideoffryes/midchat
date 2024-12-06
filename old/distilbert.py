from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer, pipeline, logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

torch.set_num_threads(32)
BATCH_SIZE = 8

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
config = DistilBertConfig.from_pretrained("distilbert-base-cased", output_hidden_states=True)
model = DistilBertModel.from_pretrained("distilbert-base-cased", config=config)

generator = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")
logging.set_verbosity_error()

memory = "You work at the United States Naval Academy, and you specialize in answering questions about the rules and regulations. Carefully consider the following information from the Midshipman Regulations Manual and base your answer off of it. Cite the passage you got the information from in your answer.\n"

def load_text(path: str, max_len=100) -> list:
    with open(path, 'r') as file:
        text = file.read()
        text = text.split(" ")
        chunks = [" ".join(text[i:i+max_len]) for i in range(0, len(text), max_len)]

    return chunks

def gen_embeds(chunks: list) -> list:
    embeddings = []
    inputs = tokenizer(chunks, padding="max_length", max_length=512, return_tensors="pt")
    input_ids_batches = torch.split(inputs["input_ids"], BATCH_SIZE)
    attention_mask_batches = torch.split(inputs["attention_mask"], BATCH_SIZE)
    for input_id, attention_mask in zip(input_ids_batches, attention_mask_batches):
        batch_inputs = {"input_ids": input_id, "attention_mask": attention_mask}
        outputs = model(**batch_inputs)
        batch_embeddings = [outputs.last_hidden_state[i][0].detach().numpy() for i in range(len(outputs.last_hidden_state))]
        embeddings += batch_embeddings
    return embeddings

def find_most_rel(query, chunks, top_k=3):
    sims = []
    query_embed = np.array(query).reshape(1, -1)
    
    for c in chunks:
        chunk_embed = np.array(c).reshape(1, -1)
        similarity = cosine_similarity(query_embed, chunk_embed)[0][0]
        sims.append(similarity)

    top_k_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
    return top_k_indices

def clean_answer(text):
    text = text.split("\n")[0]
    
    period_index = text.rfind(".")
    exc_index = text.rfind("!")
    q_index = text.rfind("?")

    max_index = max(period_index, exc_index, q_index)

    if max_index == period_index:
        return text.rsplit(".")[0] + "."
    elif max_index == exc_index:
        return text.rsplit("!")[0] + "!"
    elif max_index == q_index:
        return text.rsplit("?")[0] + "?"
    else:
        return text

def gen_answer(rel_chunks: list, query: str) -> str:
    global memory
    context = " ".join(rel_chunks)
    prompt = f"Read the following context carefully and use it to answer the question at the end of the prompt: {context}\nAnswer the following prompt: {query}\n"
    memory += prompt
    answer = generator(memory, max_new_tokens=75)[0]['generated_text'][len(memory):]
    memory += f"Answer: {answer}\n"
    return answer

def rag(query: str) -> str:
    chunks = load_text("data/midregs.txt")
    chunk_embeds = gen_embeds(chunks)
    query_embed = gen_embeds([query])[0]
    top_k_indices = find_most_rel(query_embed, chunk_embeds)
    rel_chunks = [chunks[i] for i in top_k_indices]
    answer = gen_answer(rel_chunks, query)
    return answer

if __name__ == "__main__":
    user_query = input("Input> ")
    while "quit" not in user_query.lower():
        print(f"\nAnswer> {rag(user_query)}\n")
        user_query = input("Input> ")