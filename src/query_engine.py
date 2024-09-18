from llama_index.llms.ollama import Ollama

def setup_query_engine(index):
    llm = Ollama(model="llama3.1", request_timeout=420.0)
    
    custom_prompt = (
        "You are an AI assistant specialized in analyzing automobile insurance claims data. "
        "Use the following metadata and column information to provide accurate and detailed responses:\n"
        "{metadata_and_columns}\n\n"
        "Human: {query_str}\n"
        "AI: "
    )
    
    query_engine = index.as_query_engine(
        llm=llm,
        text_qa_template=custom_prompt,
    )
    return query_engine

def see_response(query_engine, query_str):
    response = query_engine.query(query_str)
    print(response)
