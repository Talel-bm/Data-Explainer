# Insurance Claims Analyzer

## Introduction

The Insurance Claims Analyzer is a tool for processing and querying automobile insurance claims data. It combines natural language processing and database techniques to allow users to interact with structured CSV data using plain language queries.

Main features:

- CSV data processing with date parsing
- Vector embeddings for text data using HuggingFace
- Vector storage using ChromaDB for efficient searching
- Query processing using the Llama 3.1 model via Ollama
- Text-to-SQL conversion for database querying
- Command-line interface for user interaction

Built on the LlamaIndex framework, this tool is designed to help insurance professionals and analysts extract information from claim datasets using natural language queries.

## Technical Overview

- Data Ingestion: Custom CSV reader
- Embedding Model: HuggingFace
- Vector Store: ChromaDB
- Language Model: Llama 3.1 (via Ollama)
- Framework: LlamaIndex

This tool processes CSV files, converts text to vector embeddings, stores these in a vector database, and uses a language model to interpret user queries and retrieve relevant information.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/insurance-claims-analyzer.git
   cd insurance-claims-analyzer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the `config.py` file with your specific settings.

## Usage

Run the main script to start the interactive query interface:

```
python main.py
```

Enter your queries when prompted. Type 'exit' to quit the program.

## Project Structure

- `main.py`: The entry point of the application
- `config.py`: Configuration settings
- `src/`: Source code directory
  - `data_loader.py`: Custom CSV data loading functionality
  - `document_processor.py`: Document enhancement with metadata
  - `vector_store.py`: Vector store setup and management
  - `query_engine.py`: Query engine setup and execution
  - `utils.py`: Utility functions
- `tests/`: Directory for test files

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
