import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv
load_dotenv()         
embedding = OpenAIEmbeddings()   

# Folder z oczyszczonymi plikami
folder = "cleaned"

# Mapa: plik → kolekcja
pliki_i_kolekcje = {
    "Reppz_stan_na_2025-06-05.csv": "srodowisko",
    "Podsektor_centralny_2025_04.xlsx": "finanse_publiczne",
    "Ubezpieczenia_społeczne_2025_04.xlsx": "ubezpieczenia_spoleczne",
    "Dlug_publiczny_Public_debt_zMSCMsE.csv": "finanse_publiczne",
    "2024_raport_roczny_zad_1.1-Baza_danych_o_indukowanych_zjawiskach_sejsmicznych_w_GZW_w_2024_roku_od_Mw1.5p.csv": "geologia_sejsmika",
    "Baza_danych_zapadliska_LEY7UzE.csv": "geologia_sejsmika",
    "Rejestr_zastosowanie.csv": "infrastruktura"
}

#Połączenie z lokalnym Qdrant
client = QdrantClient(host="localhost", port=6333)
embedding = OpenAIEmbeddings()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

#Przetwarzanie
for plik, kolekcja in pliki_i_kolekcje.items():
    sciezka = os.path.join(folder, plik)
    print(f"\nPrzetwarzanie: {plik} → kolekcja: {kolekcja}")

    try:
        # Wczytanie danych
        if plik.endswith(".csv"):
            df = pd.read_csv(sciezka, encoding="utf-8", on_bad_lines="skip")
        elif plik.endswith(".xlsx"):
            df = pd.read_excel(sciezka, engine="openpyxl")
        else:
            print(f"Pomijam nieobsługiwany format: {plik}")
            continue


        if plik == "Rejestr_zastosowanie.csv":
            df_subset = df.head(300)
        else:
            df_subset = df

        print(f"[👁] Zawartość (do 10 wierszy):\n{df_subset.head(10).to_string(index=False)}")

        dokumenty = [
            Document(
                page_content=" | ".join(map(str, row.values)),
                metadata={"plik": plik}
            )
            for _, row in df_subset.iterrows()
        ]
        dokumenty_podzielone = splitter.split_documents(dokumenty)

        # Tworzenie kolekcji, jeśli nie istnieje
        if not client.collection_exists(kolekcja):
            client.create_collection(
                collection_name=kolekcja,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
            )

        # Inicjalizacja bazy wektorowej
        db = Qdrant(
            client=client,
            collection_name=kolekcja,
            embeddings=embedding
        )

        # Zapis dokumentów
        db.add_documents(dokumenty_podzielone)

        print(f"Kolekcja '{kolekcja}' została uzupełniona.")

    except Exception as e:
        print(f"Błąd w pliku {plik}: {e}")

print("\nGotowe! Wszystkie kolekcje zostały przetworzone.")
