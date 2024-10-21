import random
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker 
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from pytesseract import pytesseract
from typing import Any
loaders = [PyPDFLoader('./Competent_Program_Evolution.pdf')]

docs = []

# for file in loaders:
#     docs.extend(file.load())
pytesseract.tesseract_cmd = r'C:\Users\felek\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
def parse_pdf(filePath:str):
    raw_pdf_elements = partition_pdf(
    filename=filePath,
    # Unstructured first finds embedded image blocks
    extract_images_in_pdf=False,
    # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
    # Titles are any sub-section of the document
    infer_table_structure=True,
    # Post processing to aggregate text once we have the title
    chunking_strategy="by_title",
    # Chunking params to aggregate text blocks
    # Attempt to create a new chunk 3800 chars
    # Attempt to keep chunks > 2000 chars
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path='.',
)
    return raw_pdf_elements
def get_Docs():
    raw_pdf_elements = parse_pdf('./Moses.pdf')
    class Element(BaseModel):
        type: str
        text: Any
    print(raw_pdf_elements)
    # print(raw_pdf_elements[random.randint(0,len(raw_pdf_elements)-1)].page_content)
    # extract table and textual objects from parser

# Categorize by type
    categorized_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element)))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))

# Tables
    table_elements = [e for e in categorized_elements if e.type == "table"]
    print(len(table_elements))

# Text
    text_elements = [e for e in categorized_elements if e.type == "text"]
    print(len(text_elements))
    return text_elements
def test ():
    result = get_Docs()
    print(type(result[0]))
    print(type(result))
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    text_splitter = SemanticChunker(embedding_function,breakpoint_threshold_type="percentile")
    docs = [e.text for e in result] 
    docs = [text_splitter.split_text(text) for text in docs]
    texts = []
    for doc in docs:
        for text in doc:
            texts.append(text)

    vectorstore = Chroma.from_texts(texts, embedding_function, persist_directory="./moses_chroma_db")
    print(type(docs))
    print(type(docs[0]))
    print(docs[0])
    print(len(docs))
    print(vectorstore._collection.count())
    return docs
    
    


test()
# print(test())
# docs = test()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# print(type(docs[0]))


# print(vectorstore._collection.count())
# print(docs[0])
# print(type(docs))