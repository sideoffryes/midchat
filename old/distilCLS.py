from transformers import DistilBertModel, DistilBertTokenizer, pipeline, logging, AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity
import time
import os

torch.set_num_threads(32)

# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
# model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# trying new embeddings
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")

generator = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")
logging.set_verbosity_error()

memory = "You work at the United States Naval Academy, and you specialize in answering questions about the rules and regulations. Carefully consider the context given for each prompt when thinking about your answer.\n"

def load_text(path: str, max_len=100) -> list:
    with open(path, 'r') as file:
        text = file.read()
        text = text.split(" ")
        chunks = [" ".join(text[i:i+max_len]) for i in range(0, len(text), max_len)]

    return chunks

def gen_embeds(text: str):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

def find_most_rel(query, chunks, top_k=2):
    sims = []
    
    for c in chunks:
        similarity = cosine_similarity(query.squeeze(), c.squeeze(), dim=0).item()
        sims.append(similarity)

    top_k_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
    return top_k_indices

def clean_answer(text: str) -> str:
    text = text.rsplit("\n")[0]
    
    if "Answer> " not in text[:len("Answer> ")] or "Answer: " not in text[:len("Answer: ")]:
        text = "Answer> " + text
    
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
    answer = clean_answer(answer)
    memory += f"{answer}\n"
    return answer

def cache(chunks, fname="cache.pt"):
    if os.path.exists(fname):
        chunk_embeds = torch.load(fname)
    else:
        t_start = time.time()
        chunk_embeds = [gen_embeds(c) for c in chunks]
        t_stop = time.time()
        t_chunks = t_stop - t_start
        print(f"Generate chunk embeddings: {t_chunks}")
        torch.save(chunk_embeds, fname)
    return chunk_embeds

def rag(chunks: list, query: str) -> str:
    chunk_embeds = cache(chunks)
    query_embed = gen_embeds(query)
    top_k_indices = find_most_rel(query_embed, chunk_embeds)
    rel_chunks = [chunks[i] for i in top_k_indices]
    
    # DEBUG
    print(f"-----CHUNKS IDENTIFIED FOR CONTEXT-----")
    counter = 1
    for c in rel_chunks:
        print(f"{counter}. {c}\n")
        counter += 1
    
    t_start = time.time()
    answer = gen_answer(rel_chunks, query)
    t_stop = time.time()
    t_gen = t_stop - t_start
    
    print(f"----- Runtimes -----\nGenerating Llama output: {t_gen}")
    
    return answer

if __name__ == "__main__":
    chunks = load_text("data/midregs.txt")
    user_query = input("Input> ")
    while "quit" not in user_query.lower():
        print(f"\n{rag(chunks, user_query)}\n")
        user_query = input("Input> ")