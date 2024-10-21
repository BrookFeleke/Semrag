import os
import signal
import sys
import markdown
from dotenv import load_dotenv
from flask import Flask, request, render_template_string
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
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
def generate_rag_prompt(query, context):
    escaped = context.replace("'","").replace('"', "").replace("\n"," ")
    prompt = ("""
  You are a helpful and informative bot that answers questions using text from the reference context included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  You are talking to a software developer that is trying to understand a thesis paper and has question about it. \
  Format any code sections using markdown synax. \
  After specifying in markdown format what language the code will be, always start writing the code on a new line. \
  If there is code in your output, make sure to explain it in detail. \
  Your goal is to clearly explain the underlying concepts only in the context provided. \
  Your entire output should be markdown formatted, use titles and bullet points. \
  Use new lines to separate different sections. \
  If the context is irrelevant to the answer, you may ignore it.
                QUESTION: '{query}'
                CONTEXT: '{context}'
              
              ANSWER:
              """).format(query=query, context=context)
    return prompt
def get_relevant_context_from_db(query):
    context = ""
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db_2", embedding_function=embedding_function)
    search_results = vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    return context
def generate_answer(prompt):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name='gemini-pro')
    
    try:
        answer = model.generate_content(prompt)
        # Check if the response contains valid text
        if answer.text:
            return answer.text
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
        context = get_relevant_context_from_db(query)
        prompt = generate_rag_prompt(query=query, context=context)
        raw_answer = generate_answer(prompt=prompt)
        
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
        context = get_relevant_context_from_db(query)
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
# while True:
#     print("-----------------------------------------------------------------------\n")
#     print("What would you like to ask?")
#     query = input("Query: ")
#     context = get_relevant_context_from_db(query)
#     prompt = generate_rag_prompt(query=query, context=context)
#     answer = generate_answer(prompt=prompt)
#     print(answer)