import os
import signal
import sys
from chromadb import Client
import markdown
from dotenv import load_dotenv
from flask import Flask, request, render_template_string
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from nltk.corpus import wordnet
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
# welcome_text = generate_answer("Can you quickly introduce yourself")
app = Flask(__name__)
# Load environment variables from .env file
load_dotenv()
# Access the API key from environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
def signal_handler(sig, frame):
    print('\nThanks for using Gemini. :)')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
def expand_query(query):
    """
    Function to augment a query by generating a hypothetical answer 
    using the Gemini LLM and appending it to the original query.
    """
    context = """Competent Program Evolution - this is the central focus of the document, and likely to be a key concept for users to understand.
Representation-Building - this is a crucial process in competent program evolution, involving transforming the program space to facilitate effective search.
Program Spaces - the document discusses the properties and structure of program spaces in detail, which is important for understanding how evolution works.
Boolean Formulae - these are used extensively as a model system throughout the document to illustrate key concepts.
Hierarchical Normal Forms - these are special representations of programs that are designed to reduce redundancy and highlight structure, making them easier to evolve.
Problem Difficulty - the document discusses a novel bipartite model of problem difficulty for program evolution, which is important for understanding the limits and capabilities of these systems.
Meta-Optimizing Semantic Evolutionary Search (MOSES) - this is the name of the specific framework for competent program evolution presented in the document.
Hierarchical Bayesian Optimization Algorithm (hBOA) - this is a powerful optimisation algorithm that is used within MOSES.
Deme - this is a term used to describe a population of programs within a particular region of program space defined by a specific representation.
Metapopulation - a set of demes spanning an area within program space.
Supervised Classification - the document evaluates MOSES on real-world supervised classification problems, demonstrating its applicability to practical tasks.
Genetic Programming - this is a related approach to program evolution that is often compared to MOSES.
Open-endedness, Over-representation, Compositional Hierarchy - these are some of the key properties of program spaces that make them difficult to evolve.
Behavioural Decomposability, White Box Execution - these are properties that MOSES leverages to facilitate evolution.
Syntactic Distance, Semantic Distance - these are important concepts for understanding how representation-building affects the structure of program space.
Bloat - this is a common phenomenon in program evolution where programs tend to become unnecessarily large, which MOSES is designed to mitigate."""
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name='gemini-pro')
    messages = [
    {
        "role": "model",
        "parts": [
            "You are a highly intelligent prompt engineer designed to expand and optimize user queries for better performance with large language models (LLMs) and vector similarity search systems like ChromaDB.",
            "Your task is to reword and enhance the query provided within the delimiters #^query^# while ensuring it is relevant to the core concepts of the document described in the context.",
            "The document focuses on Competent Program Evolution, and related concepts include representation-building, program spaces, Boolean formulae, hierarchical normal forms, and Meta-Optimizing Semantic Evolutionary Search (MOSES).",
            "It is crucial that the expanded query remains within the context of the key terms and concepts provided, avoiding unrelated subjects such as religion, mythology, or anything outside the domain of program evolution, optimization, or machine learning.",
            "Use only the context provided to rephrase the query. Do not introduce information that is not grounded in the context, and ensure the reworded query is well-suited for vector similarity search by including relevant keywords that enhance the precision of search results.",
            "If the original query lacks details or clarity, add relevant elements that maintain the meaning while improving its searchability and clarity for both LLMs and the database.",
            "Your main goal is to add relevant keywords to the query to enhance its relevance to the document, and ensure it is well-suited for vector similarity search by including relevant keywords that enhance the precision of search results.",
            "What you return should be only the enhanced query as a string and nothing else."
        ]
    },
    {
        "role": "user",
        "parts": [
            f"Query: #^{query}^#",
            f"Context: #^{context}^#"
        ]
    }
]
    try:
        # Generating a response using the Gemini LLM
        response = model.generate_content(
            messages
        )
        return response.candidates[0].content.parts[0].text.replace("#^", "").replace("^#", "")
    except Exception as e:
        return f"Error generating query: {str(e)}"

# Reranking function based on cosine similarity
def get_relevant_context_from_db(query):
    context = ""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vector_db = Chroma(persist_directory="./comprog_chroma_db", embedding_function=embedding_model)

    # Perform initial search with a larger k value
    initial_search_results = vector_db.similarity_search(query, k=20)

    # Re-rank results using cosine similarity
    query_embedding = embedding_model.embed_query(query)
    query_embedding = np.array(query_embedding).reshape(1,-1)
    reranked_results = []
    for result in tqdm(initial_search_results, desc="Re-ranking results"):
        result_embedding = embedding_model.embed_query(result.page_content)
        result_embedding = np.array(result_embedding).reshape(1,-1)
        c_similarity = cosine_similarity(query_embedding, result_embedding)
        reranked_results.append((result, c_similarity[0][0]))

    # Sort results by cosine similarity in descending order
    reranked_results.sort(key=lambda x: x[1], reverse=True)
    # Take the top k results
    k = 10
    for result, _ in reranked_results[:k]:
        context += result.page_content + "\n"

    return context
def generate_answer(query,context):
    """
    Uses the Gemini API to generate a response based on the context provided.

    Args:
        prompt (str): The prompt to generate a response to.

    Returns:
        str: The generated response. If the response is invalid or blocked, an error message is returned.
    """
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name='gemini-pro')
    messages = [
    {
        "role": "model",
        "parts": [
            "You are a highly intelligent and competent research assistant tasked with answering specific research-related questions based strictly on the provided context.",
            "You are given a query and relevant context, and you must answer only the question specified within the delimiters #^query^#.",
            "Use only the information found in the context provided within the delimiters #^context^#.",
            "Do not provide information outside of the context, and if the context does not contain the necessary information to answer the question, respond by stating that the information is not available.",
            "Format your response using markdown syntax. Include clear section titles and, where applicable, use bullet points to organize information.",
            "Your goal is to ensure factual accuracy, minimize hallucinations, and only provide answers grounded in the context.",
            "Focus on providing a clear, concise, and relevant response to the query, without adding unnecessary details or summarizing unrelated portions of the document.",
            "Your output should only contain the answer and not the question."
        ]
    },
    {
        "role": "user",
        "parts": [
            f"Query: #^^{query}^^#",
            f"Context: #^^{context}^^#"
        ]
    }
]
    try:
        answer = model.generate_content(messages)
        # Check if the response contains valid text
        if answer.candidates:
            return answer.candidates[0].content.parts[0].text.replace("#^^", "").replace("^^#", "")
        else:
            return "Sorry, the response was blocked due to safety concerns."
    except ValueError as e:
        # Handle the case where the response is invalid or blocked
        return f"An error occurred: {str(e)}. Please try again or rephrase your query."
# welcome_text = generate_answer("Can you quickly introduce yourself")
# print(welcome_text)
@app.route('/', methods=['GET', 'POST'])
def index():
    answer = ""
    query = ""
    if request.method == 'POST':
        query = request.form['query']
        expanded_query = expand_query(query)
        context = get_relevant_context_from_db(expanded_query)
        # prompt = generate_rag_prompt(query=expanded_query, context=context)
        raw_answer = generate_answer(expanded_query,context)
        print("-------------------------answer------------------")
        print(raw_answer)
        # Convert the answer from markdown to HTML
        answer = markdown.markdown(raw_answer)

    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Gemini AI Q&A</title>
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
            <style>
                .markdown-content {
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                    white-space: pre-wrap;
                    max-height: 300px; /* Limit height */
                    overflow-y: auto;  /* Make it scrollable */
                    padding-right: 10px; /* Add some padding for better UX */
                }
            </style>
        </head>
        <body class="bg-gray-100 flex justify-center items-center h-screen">
            <div class="bg-white p-8 rounded shadow-md w-full max-w-5xl" style="width: 70%;">
                <h1 class="text-3xl font-bold mb-8 text-center">Ask a Question</h1>
                <form method="POST" class="space-y-4">
                    <div>
                        <label for="query" class="block text-lg font-medium text-gray-700">Query:</label>
                        <input placeholder="{{ query }}" type="text" id="query" name="query" class="mt-1 block w-full px-4 py-3 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-green-500 focus:border-green-500 sm:text-lg" required>
                    </div>
                    <div class="flex justify-center">
                        <input type="submit" value="Submit" class="bg-green-600 text-white py-2 px-6 rounded-md hover:bg-green-700">
                    </div>
                </form>
                {% if answer %}
                <div class="mt-10">
                    <h2 class="text-2xl font-semibold mb-4 text-green-700">Answer:</h2>
                    <div class="text-gray-800 text-lg markdown-content">{{ answer | safe }}</div>
                </div>
                {% endif %}
                <div class="mt-4">
                    <a href="/view-fetched-text?query={{ query }}" class="text-blue-600 hover:underline">View Fetched Text from ChromaDB</a>
                </div>
            </div>
        </body>
        </html>
    ''', answer=answer, query=query)
@app.route('/view-fetched-text')
def view_fetched_text():
    query = request.args.get('query', '')
    if query:
        expanded_query = expand_query(query)
        context = get_relevant_context_from_db(expanded_query)
    else:
        context = "No query provided."
    
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Fetched Text</title>
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
            <style>
                .markdown-content {
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                    white-space: pre-wrap;
                    max-height: 300px; /* Limit height */
                    overflow-y: auto;  /* Make it scrollable */
                    padding-right: 10px; /* Add some padding for better UX */
                }
            </style>
        </head>
        <body class="bg-gray-100 flex justify-center items-center h-screen">
            <div class="bg-white p-8 rounded shadow-md w-full max-w-5xl" style="width: 70%;">
                <h1 class="text-3xl font-bold mb-8 text-center">Fetched Text from ChromaDB</h1>
                <div class="mt-10">
                <h2 class="text-2xl font-semibold mb-4 text-green-700">Answer:</h2>
                <div class="text-gray-800 text-lg markdown-content">{{ context }}</div>
                </div>                
                <div class="mt-4">
                    <a href="/" class="text-blue-600 hover:underline">Back to Main Page</a>
                </div>
            </div>
        </body>
        </html>
    ''', context=context)
if __name__ == '__main__':
    app.run(debug=True)
