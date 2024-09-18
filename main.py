import os
from src.vector_store import setup_vector_store
from src.query_engine import setup_query_engine, query_to_csv
from config import CSV_FOLDER

def main():
    # Set up the vector store
    index = setup_vector_store(CSV_FOLDER)

    # Set up the query engine
    query_engine = setup_query_engine(index)

    # Interactive query loop
    while True:
        query_str = input("Enter your query (or 'exit' to quit): ")
        if query_str.lower() == 'exit':
            break

        query_to_csv(query_engine, query_str)

if __name__ == "__main__":
    main()
