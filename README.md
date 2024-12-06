# MIDCHAT

## A USNA MIDREGs Chatbot

### by Henry Frye

## About

MIDCHAT is an AI-powered chatbot that is designed to answer questions specifically about the U.S. Naval Academy. The chatbot has access to hundreds of USNA instructions and notices that it can pull information from to answer the users's questions.

## How it works

MIDCHAT implements a retrieval augmented generation (RAG) approach by adding snippets of information from external sources (USNA instructions and notices) as extra context to a large language model (LLM) to answer the users' questions. In this case, this project uses Facebook AI Similarity Search (FAISS) to identify similar snippets of data to the user's query and feeds that data to Meta's Llama 3.2 3B Instruct model to answer questions and create an interactive chat session.
