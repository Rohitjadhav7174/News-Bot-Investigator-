import os
import pickle
import time
import tkinter as tk
from tkinter import messagebox, simpledialog
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

class NewsBotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("News Bot Investigator Tool ")

        self.urls = []
        self.file_path = "faiss_store_openai.pkl"
        self.llm = OpenAI(temperature=0.9, max_tokens=500)

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="News Article URLs", font=("Helvetica", 16)).pack(pady=10)
        
        self.url_entries = []
        for i in range(3):
            frame = tk.Frame(self.root)
            frame.pack(pady=5)
            tk.Label(frame, text=f"URL {i+1}:", font=("Helvetica", 12)).pack(side=tk.LEFT)
            entry = tk.Entry(frame, width=50)
            entry.pack(side=tk.LEFT)
            self.url_entries.append(entry)

        self.process_button = tk.Button(self.root, text="Process URLs", command=self.process_urls)
        self.process_button.pack(pady=20)

        self.query_label = tk.Label(self.root, text="Question:", font=("Helvetica", 14))
        self.query_label.pack(pady=5)
        self.query_entry = tk.Entry(self.root, width=60)
        self.query_entry.pack(pady=5)
        
        self.query_button = tk.Button(self.root, text="Get Answer", command=self.get_answer)
        self.query_button.pack(pady=20)

        self.answer_label = tk.Label(self.root, text="", font=("Helvetica", 12), wraplength=600, justify=tk.LEFT)
        self.answer_label.pack(pady=10)

        self.sources_label = tk.Label(self.root, text="", font=("Helvetica", 12), wraplength=600, justify=tk.LEFT)
        self.sources_label.pack(pady=10)

    def process_urls(self):
        urls = [entry.get() for entry in self.url_entries]
        if any(urls):
            self.answer_label.config(text="Data Loading...Started...")
            self.root.update_idletasks()
            
            # load data
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            self.answer_label.config(text="Text Splitter...Started...")
            self.root.update_idletasks()
            docs = text_splitter.split_documents(data)

            embeddings = OpenAIEmbeddings()
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            self.answer_label.config(text="Embedding Vector Started Building...")
            self.root.update_idletasks()
            time.sleep(2)

            with open(self.file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)
            
            messagebox.showinfo("Success", "URLs processed successfully!")
            self.answer_label.config(text="")
        else:
            messagebox.showwarning("Input Error", "Please enter at least one URL.")

    def get_answer(self):
        query = self.query_entry.get()
        if query:
            if os.path.exists(self.file_path):
                with open(self.file_path, "rb") as f:
                    vectorstore = pickle.load(f)
                    chain = RetrievalQAWithSourcesChain.from_llm(llm=self.llm, retriever=vectorstore.as_retriever())
                    result = chain({"question": query}, return_only_outputs=True)

                    self.answer_label.config(text=f"Answer: {result['answer']}")

                    sources = result.get("sources", "")
                    if sources:
                        sources_list = sources.split("\n")
                        sources_text = "Sources:\n" + "\n".join(sources_list)
                        self.sources_label.config(text=sources_text)
                    else:
                        self.sources_label.config(text="")

            else:
                messagebox.showwarning("Error", "Vectorstore file not found. Please process URLs first.")
        else:
            messagebox.showwarning("Input Error", "Please enter a question.")

if __name__ == "__main__":
    root = tk.Tk()
    app = NewsBotApp(root)
    root.mainloop()
