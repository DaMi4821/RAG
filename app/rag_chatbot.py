import os
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()  

opisy = {
    "srodowisko": "Rejestr przedstawicielstw zagranicznych przedsiębiorstw działających na terenie Polski — zawiera dane o nazwie firmy, kraju pochodzenia, osobach reprezentujących, branży działalności oraz lokalizacji przedstawicielstw.",
    "finanse_publiczne": "Zawiera dane o stanie zadłużenia publicznego (krajowego i zagranicznego), strukturze długu oraz finansach sektora instytucji rządowych i samorządowych, w tym podsektora centralnego.",
    "ubezpieczenia_spoleczne": "Zawiera statystyki dotyczące systemu ubezpieczeń społecznych w Polsce, obejmujące liczbę osób objętych ubezpieczeniem, składki, wypłaty świadczeń oraz ich strukturę według różnych kryteriów (np. grup zawodowych).",
    "geologia_sejsmika": "Baza danych dotycząca zjawisk sejsmicznych oraz zapadlisk na obszarze Górnośląskiego Zagłębia Węglowego — obejmuje daty, lokalizacje, magnitudy trzęsień oraz charakterystykę geologiczną incydentów.",
    "infrastruktura": "Zestawienie technologii, urządzeń i rozwiązań stosowanych w różnych sektorach gospodarki — prezentuje dane o zastosowaniach, obszarach wdrożeń oraz potencjalnych zastosowaniach infrastrukturalnych."
}

#LLM + embedding + Qdrant
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
embedding = OpenAIEmbeddings()
client = QdrantClient(host="localhost", port=6333)


collection_prompt = PromptTemplate.from_template(
    """Twoim zadaniem jest wybrać najbardziej odpowiednią kolekcję danych spośród dostępnych,
     w oparciu o zapytanie użytkownika i poniższe opisy:
{descriptions}

Na podstawie pytania: "{query}", zwróć **tylko nazwę kolekcji** (np. 'geologia_sejsmika')."""
)
select_chain = collection_prompt | llm


chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="Jesteś uprzejmym i rzeczowym urzędnikiem państwowym. "
                          "Twoim obowiązkiem jest udzielanie kulturalnych, konkretnych i precyzyjnych odpowiedzi "
                          "obywatelom na podstawie dostępnych danych. "
                          "Jeśli nie posiadasz wystarczających informacji, jasno to zakomunikuj."),
    ("human", """
Odpowiadaj wyłącznie na podstawie dostarczonych danych kontekstowych.
Nie zgaduj. Jeśli kontekst nie zawiera odpowiedzi, napisz:
"Nie posiadam wystarczających informacji, aby odpowiedzieć na to pytanie."

Jeśli pytanie jest zbyt ogólne, napisz:
"Proszę zadać bardziej szczegółowe pytanie."

Kontekst:
{context}

Pytanie:
{question}
""")
])


print("Możesz zadawać pytania. Wpisz 'exit', aby zakończyć.\n")

MAX_ATTEMPTS = 3

while True:
    user_query = input("Twoje pytanie: ")
    if user_query.lower() in ["exit", "quit"]:
        print("Zakończono.")
        break

    desc_text = "\n".join([f"{key}: {value}" for key, value in opisy.items()])
    collection_choice = select_chain.invoke({
        "descriptions": desc_text,
        "query": user_query
    }).content.strip()

    if collection_choice not in opisy:
        print("Nie udało się rozpoznać odpowiedniej kolekcji. Spróbuj inaczej sformułować pytanie.")
        continue

    vectordb = Qdrant(
        client=client,
        collection_name=collection_choice,
        embeddings=embedding
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": chat_prompt},
        return_source_documents=True
    )

    result = None
    for attempt in range(MAX_ATTEMPTS):
        print(f"Próba {attempt + 1}...")
        attempt_result = qa_chain.invoke({"query": user_query})
        answer = attempt_result["result"]

        if "nie posiadam wystarczających informacji" not in answer.lower():
            result = attempt_result
            break
        elif attempt == MAX_ATTEMPTS - 1:
            result = attempt_result

    print(f"\n[Kolekcja: {collection_choice}]")
    print("Odpowiedź urzędnika:\n", result["result"])

    print("\nŹródła:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("plik", "[brak informacji]"))
    print("-" * 60)
