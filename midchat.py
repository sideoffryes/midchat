import argparse
import os
import re
import time

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, logging, pipeline
from utils import readSources

# parse commmand line arguments
parser = argparse.ArgumentParser(description="Answer questions about MIDREGs from the US Naval Academy using Llama 3.2-1B Instruct and RAG!\n\nThis script will automatically use a GPU if one is available.")
parser.add_argument("-m", "--max-tokens", type=int, help="Specify the maximum number of output tokens when giving a response, default is 75.", default=75)
parser.add_argument("-t", "--num-threads", type=int, help="Specify the number of threads to run the process, default is 16.", default=16)
parser.add_argument("-r", "--regen-cache", action="store_true", help="Manually force the script to regenerate the FAISS index and word vector embedding caches.")
parser.add_argument("-k", "--top-k", type=int, help="Specify the number of related passages to identify from MIDREGs to use for context when answering the prompt, default is 5.", default=5)
parser.add_argument("-v", "--verbose", action="store_true", help="Enables extra debugging information.")
parser.add_argument("-c", "--cpu", action="store_true", help="Force CPU mode even if GPU is available")
args = parser.parse_args()

torch.set_num_threads(args.num_threads)

# force CPU based on args or check for GPU
if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# trying new embeddings
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large", device_map="auto")

generator = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct", device_map="auto")
logging.set_verbosity_error()

# Set constant memory with role, task, and rules
memory = f"Role: You work at the United States Naval Academy, and you specialize in answering questions about the rules and regulations. All of your answers are related to Naval Academy Midshipmen. You are always confident in your answers.\n"
memory += f"Task: Carefully analyze each of the passages that you are given from MIDREGs as context to help you answer the question.\n"
memory += f"Rules: Not all of the context that you will be given will be correct. Do not repeat the exact context when you give your asnwer. Answer the question in at most two sentences. Your answer must end in a complete sentence. Do not repeat yourself. Do not provide any follow-up questions or answers.\n\n"

# few-shot prompting
memory += f"Question: When is end of liberty formation?\nAnswer: End of Liberty Formation typically occurs on Sunday at 1800.\n\n"
memory += f"Question: Can midshipmen drink in Bancroft Hall?\nAnswer: No. Midshipmen are never allowed to drink in Bancroft Hall.\n\n"
memory += f"Question: Can a 4/C wear civvies?\nAnswer: No. A 4/C cannot wear civilian attire unless they are on leave away from the Naval Academy.\n\n"
memory += f"Question: Can I talk on the phone when I walk in uniform?\nAnswer: Yes. You may talk on the phone while walking in uniform as long as you are able to lower the phone and render a salute.\n\n"
memory += f"Question: Who can wear civvies on liberty?\nAnswer: 1/C and 2/C midshipmen may wear civilian attire while on town liberty.\n\n"

def cache_faiss(chunks, fname="cache.faiss"):
    index_file = fname
    embed_file = fname.replace(".faiss", ".npy")
    
    if os.path.exists(index_file) and os.path.exists(embed_file) and not args.regen_cache:
        index = faiss.read_index(index_file)
        chunk_embeds = np.load(embed_file)
    else:
        chunk_embeds = [gen_embeds(c).cpu().detach().numpy().flatten() for c in chunks]
        chunk_embeds = np.vstack(chunk_embeds)
        
        # normalize for cosine similarity
        faiss.normalize_L2(chunk_embeds)
        
        # create and train faiss index
        index = faiss.IndexFlatIP(chunk_embeds.shape[1])
        index.add(chunk_embeds)
        
        faiss.write_index(index, index_file)
        np.save(embed_file, chunk_embeds)
    
    return index, chunk_embeds

def load_text() -> list:
    files = readSources()
    chunks = []
    
    for f in files:
        with open(f, 'r') as file:
            text = file.read()
            chunks += text.split("\n\n")

    return chunks

def gen_embeds(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

def find_most_rel(query, index):
    query_embed = gen_embeds(query).cpu().detach().numpy().flatten().reshape(1, -1)
    faiss.normalize_L2(query_embed)
    _, top_k_indices = index.search(query_embed, args.top_k)
    return top_k_indices[0]

def clean_answer(text: str) -> str:
    text = text.replace("Answer: ", "")
    text = text.split("\n\n")[0]

    return text    

def gen_answer(rel_chunks: list, query: str) -> str:
    global memory
    
    # create temporary prompt independent from memory that uses the context identified from MIDREGs
    prompt = memory
    for c in rel_chunks:
        prompt += f"Context: {c}\n"
    
    prompt += f"Question: {query}\n"
    answer = generator(prompt, max_new_tokens=args.max_tokens)[0]['generated_text'][len(prompt):]
    answer = clean_answer(answer)
    
    # Only add question and answer to memory
    memory += f"Question: {query}\n"
    memory += f"Answer: {answer}\n"
    return answer

def rag(chunks: list, query: str) -> str:
    t_start = time.time()
    index, chunk_embeds = cache_faiss(chunks)
    t_stop = time.time()
    t_embeds = t_stop - t_start
    
    top_k_indices = find_most_rel(query, index)
    rel_chunks = [chunks[i] for i in top_k_indices]
    
    # Verbose mode
    if args.verbose:
        print(f"-----CHUNKS IDENTIFIED FOR CONTEXT-----")
        counter = 1
        for c in rel_chunks:
            print(f"{counter}. {c}\n")
            counter += 1
        
    t_start = time.time()
    answer = gen_answer(rel_chunks, query)
    t_stop = time.time()
    t_gen = t_stop - t_start
        
    if args.verbose:
        print(f"----- Runtimes -----\nGenerating chunk embeddings: {t_embeds}\nGenerating Llama output: {t_gen}")
    
    return answer

if __name__ == "__main__":
    # Load text from MIDREGs
    # Get the user's input, check for EOF or blank input
    try:
        user_query = input("Input> ")
    except EOFError:
        print("Bye!")
        quit()
        
    chunks = load_text()
    
    # Loop until user types "quit"
    while "quit" not in user_query.lower():
        print(f"\n{rag(chunks, user_query)}\n")
        try:
            user_query = input("Input> ")
        except EOFError:
            print("Bye!")
            
            # verbose mode
            if args.verbose:
                print(f"---------- MEMORY ----------\n{memory}")
            
            quit()