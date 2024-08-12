import os
import streamlit as st
import requests
import asyncio
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from datasets import load_dataset, load_metric
import torch

# Set up Hugging Face API key as an environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your huggingface api_key"

# Step 1: Data Collection and Preprocessing

async def download_books_async():
    books_urls = [
        "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride and Prejudice
        "https://www.gutenberg.org/cache/epub/11/pg11.txt",       # Alice's Adventures in Wonderland
    ]
    
    books = []
    for url in books_urls:
        response = await asyncio.to_thread(requests.get, url)
        if response.status_code == 200:
            books.append(response.text)
        else:
            st.warning(f"Failed to download book from {url}")
    
    return books

async def load_booksum_dataset_async(subset_size=1000):
    """
    Load a subset of the booksum dataset from Hugging Face asynchronously.
    """
    dataset = await asyncio.to_thread(load_dataset, "kmfoda/booksum", split="train[1:{}]".format(subset_size))
    combined_texts = []
    for example in dataset:
        text = example['chapter']
        summary = example['summary_text']
        combined_text = f"{text} Summary: {summary}"
        combined_texts.append(combined_text)
    return combined_texts

def preprocess_text(text):
    text = text.lower()
    text = ' '.join(text.split())
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Step 2: Setting Up the Retrieval System

@st.cache_resource
def create_retrieval_system(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(docs, embeddings)
    return vector_store

async def retrieve_relevant_passages(query, vector_store, top_k=3):
    results = await asyncio.to_thread(vector_store.similarity_search, query, k=top_k)
    return [doc.page_content for doc in results]

# Step 3: Load T5 Model

@st.cache_resource
def load_t5_model():
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    return tokenizer, model

@torch.no_grad()
async def generate_summary_async(input_text, tokenizer, model, max_length=150, is_augmented=False):
    if is_augmented:
        prefix = "Generate a comprehensive summary of the following text and additional context: "
        max_length = 250  # Increase max length for augmented summary
    else:
        prefix = "Summarize the following text: "
    
    input_text = prefix + input_text.strip().replace("\n", " ")
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = await asyncio.to_thread(model.generate,
                                          input_ids,
                                          num_beams=4,
                                          no_repeat_ngram_size=2,
                                          min_length=50,
                                          max_length=max_length,
                                          early_stopping=True,
                                          do_sample=True,
                                          top_k=50,
                                          temperature=0.7)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Step 4: Evaluation Metrics

async def evaluate_summary_async(predicted_summary, reference_summary):
    """
    Evaluates the generated summary using ROUGE metrics asynchronously.
    """
    rouge = await asyncio.to_thread(load_metric, "rouge", trust_remote_code=True)
    rouge_scores = await asyncio.to_thread(rouge.compute, predictions=[predicted_summary], references=[reference_summary])

    return rouge_scores

# Step 5: Streamlit Interface

async def main():
    st.title("📚 Augmented Books Summarizer")

    st.markdown("""
    This application uses a Retrieval-Augmented Generation (RAG) pipeline to produce enhanced book summaries. 
    It combines the T5-small model with a retrieval system built from selected Project Gutenberg books and the booksum dataset.
    """)
    
    tokenizer, model = load_t5_model()
    
    if 'vector_store' not in st.session_state:
        with st.spinner('Downloading and preprocessing books...'):
            # Load Gutenberg books
            books = await download_books_async()
            preprocessed_books = [preprocess_text(book) for book in books]
            passages = [chunk for book in preprocessed_books for chunk in chunk_text(book)]
            
            # Load Booksum dataset
            booksum_data = await load_booksum_dataset_async()
            preprocessed_booksum = [preprocess_text(text) for text in booksum_data]
            passages.extend(preprocessed_booksum)  # Combine both sources

            # Create the retrieval system with the combined data
            vector_store = create_retrieval_system(passages)
            st.session_state['vector_store'] = vector_store
        st.success("Books and dataset processed; retrieval system ready!")
    
    st.header("Generate an Augmented Summary")
    user_input = st.text_area("Enter the text you want to summarize:", height=150)
    
    if st.button("Generate Summary"):
        vector_store = st.session_state['vector_store']
        
        with st.spinner('Generating summaries...'):
            initial_summary = await generate_summary_async(user_input, tokenizer, model)
            relevant_passages = await retrieve_relevant_passages(user_input, vector_store)
            augmented_input = f"{user_input}\n\nAdditional context:\n" + "\n".join(relevant_passages)
            augmented_summary = await generate_summary_async(augmented_input, tokenizer, model, is_augmented=True)
            
            st.subheader("Initial Summary")
            st.write(initial_summary)
            
            st.subheader("Augmented Summary")
            st.write(augmented_summary)
            
            st.session_state['initial_summary'] = initial_summary
            st.session_state['augmented_summary'] = augmented_summary
    
    st.header("Evaluate the Summary")
    reference_summary = st.text_area("Enter the reference summary for evaluation:", height=150, key="reference_summary")
    
    if st.button("Evaluate Summaries"):
        if 'initial_summary' in st.session_state and 'augmented_summary' in st.session_state and reference_summary:
            with st.spinner('Evaluating summaries...'):
                # Evaluate Initial Summary
                rouge_scores_init = await evaluate_summary_async(st.session_state['initial_summary'], reference_summary)
                st.subheader("Initial Summary Scores")
                st.json({key: value.mid.fmeasure * 100 for key, value in rouge_scores_init.items()})

                # Evaluate Augmented Summary
                rouge_scores_aug = await evaluate_summary_async(st.session_state['augmented_summary'], reference_summary)
                st.subheader("Augmented Summary Scores")
                st.json({key: value.mid.fmeasure * 100 for key, value in rouge_scores_aug.items()})

if __name__ == "__main__":
    asyncio.run(main())
