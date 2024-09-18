# Insurance Claims Analyzer

## Introduction

The Insurance Claims Analyzer is a Natural Language Processing (NLP) and Machine Learning (ML) powered tool designed for in-depth analysis of automobile insurance claims data. This project leverages vector embedding techniques and Large Language Models (LLMs) to enable semantic search and natural language querying of structured CSV data.

Key technical features include:

- **Custom CSV Data Ingestion**: Utilizes a bespoke CSV reader with intelligent date parsing and document structuring.
- **Vector Embedding**: Implements HuggingFace embeddings to convert textual data into high-dimensional vector representations.
- **Efficient Vector Storage**: Integrates ChromaDB as a vector store for optimized similarity search capabilities.
- **LLM-Powered Query Engine**: Employs the Llama 3.1 model via Ollama for natural language understanding and query processing.
- **Interactive CLI**: Provides a command-line interface for real-time data exploration and analysis.

This tool is built on the LlamaIndex framework, offering a scalable and extensible architecture for handling large volumes of insurance claim data. It's designed to assist insurance professionals, data analysts, and researchers in extracting actionable insights from complex claim datasets through intuitive natural language interactions.

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
