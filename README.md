# RAG Chatbot for Government Data — Python + LangChain + Qdrant

This project is a practical implementation of a **Retrieval-Augmented Generation (RAG)** system based on Large Language Models (LLMs), designed to operate on real public data. It consists of two main modules:

- **`load_to_qdrant.py`** – loads CSV/XLSX data, transforms rows into embeddings, and stores them in a Qdrant vector database.
- **`rag_chatbot.py`** – a formal chatbot (styled as a public official) powered by GPT-3.5-Turbo that dynamically retrieves relevant data chunks to answer user queries.

The system enables users to **ask natural language questions in Polish** and receive **accurate, source-based answers only**. It demonstrates a production-grade example of using LLMs in environments where information traceability and reliability are crucial.

---

## 🔧 Technologies

- **LangChain** – for prompt management, chain orchestration, and integration with Qdrant.
- **OpenAI / GPT-3.5 Turbo** – LLM used for generating answers.
- **Qdrant (local)** – vector database for embedding storage and retrieval.
- **Python + Pandas** – for processing structured tabular data from CSV and Excel files.
- **OpenAI Embeddings** – for semantic encoding of tabular records into vectors.

---

## 🧠 Features

### 1. `load_to_qdrant.py` – Data ingestion to vector database

- Automatically detects `.csv` and `.xlsx` files from the `cleaned/` folder.
- Maps each file to a distinct **Qdrant collection** based on content type (e.g. `geologia_sejsmika`, `finanse_publiczne`).
- Generates embeddings for each row using **OpenAI’s `text-embedding-3-large`** model.
- Uses `CharacterTextSplitter` to chunk large rows for efficient semantic search.
- Creates Qdrant collections if they don't already exist.
- Stores each chunk as a document, tagged with metadata about the original file.

### 2. `rag_chatbot.py` – Context-aware chatbot with selective retrieval

- Accepts open-ended questions from users in natural Polish.
- Dynamically selects the most relevant data collection based on a semantic prompt (`select_chain`).
- Retrieves the top-k relevant chunks from Qdrant using `RetrievalQA`.
- Answers are generated in a **formal, precise, and official tone** — mimicking a government clerk.
- If insufficient information is found in the documents, the model responds clearly with: *"Nie posiadam wystarczających informacji..."*

---

## 📁 Project structure

```text
project/
├── app/
│   └── rag_chatbot.py          # chatbot with collection selector
├── loaders/
│   └── load_to_qdrant.py       # ingest data into Qdrant
├── cleaned/                    # folder with cleaned tabular files
├── .env                        # (local) OpenAI key and config
├── requirements.txt
└── README.md
```


# RAG Chatbot dla danych rządowych — Python + LangChain + Qdrant

Ten projekt to praktyczna implementacja systemu **RAG (Retrieval-Augmented Generation)** opartego na dużych modelach językowych (LLM), który działa na rzeczywistych danych publicznych. Składa się z dwóch głównych modułów:

- **`load_to_qdrant.py`** – ładowanie danych CSV/XLSX, przekształcanie ich w embeddingi i zapis do bazy wektorowej Qdrant.
- **`rag_chatbot.py`** – chatbot urzędowy oparty na GPT-3.5-Turbo, który selektywnie pobiera kontekst z bazy i udziela odpowiedzi na pytania użytkownika.

System umożliwia **zadawanie naturalnych pytań w języku polskim** i udzielanie precyzyjnych odpowiedzi **wyłącznie na podstawie załadowanych danych**. Idealny przykład zastosowania LLM w środowisku, gdzie ważna jest kontrola źródła informacji.

---

## 🔧 Technologie

- **LangChain** – do zarządzania łańcuchami zapytań, promptów i integracji z Qdrant.
- **OpenAI / GPT-3.5 Turbo** – model LLM do generowania odpowiedzi.
- **Qdrant (local)** – lokalna baza wektorowa do przechowywania embeddingów.
- **Python + Pandas** – przetwarzanie danych tabularnych z CSV/XLSX.
- **OpenAI Embeddings** – przekształcanie rekordów danych w wektory semantyczne.

---

## 🧠 Funkcje

### 1. `load_to_qdrant.py` – ładowanie danych do bazy wektorowej

- Automatycznie rozpoznaje pliki `.csv` i `.xlsx` w folderze `cleaned/`.
- Mapuje każdy plik do osobnej **kolekcji Qdranta** według typu danych (np. `geologia_sejsmika`, `finanse_publiczne`).
- Tworzy embeddingi dla każdego rekordu przy użyciu **OpenAI Embeddings (text-embedding-3-large)**.
- Używa `CharacterTextSplitter`, aby lepiej indeksować dane o dużej długości.
- Tworzy kolekcje w Qdrant, jeśli nie istnieją.
- Przechowuje dokumenty w postaci chunków z metadanymi o źródle.

### 2. `rag_chatbot.py` – chatbot z selektywnym kontekstem

- Użytkownik wpisuje pytanie w języku naturalnym.
- System najpierw wybiera właściwą kolekcję danych na podstawie opisu (`select_chain`).
- Następnie wykorzystuje `RetrievalQA` i Qdranta, aby pobrać najbardziej relewantne fragmenty danych (k=5).
- Chatbot udziela **formalnych, precyzyjnych i bezpiecznych odpowiedzi** (symulując urzędnika państwowego).
- Jeśli dane nie zawierają odpowiedzi, model jasno informuje: *"Nie posiadam wystarczających informacji..."*

---

## 📁 Struktura katalogów

```text
project/
├── app/
│   └── rag_chatbot.py          # chatbot z wyborem kolekcji
├── loaders/
│   └── load_to_qdrant.py       # ładowanie danych do Qdrant
├── cleaned/                    # folder z oczyszczonymi plikami danych
├── .env                        # (lokalny) klucz API
├── requirements.txt
└── README.md
```
