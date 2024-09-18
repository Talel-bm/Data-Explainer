# Insurance Claims Analyzer

This project is an AI-powered tool for analyzing automobile insurance claims data. It uses advanced natural language processing techniques to query and analyze CSV data files containing insurance claim information.

## Features

- Custom CSV data loader with date parsing
- Integration with Chroma vector store for efficient data querying
- Query engine powered by the Llama 3.1 model via Ollama
- Interactive command-line interface for querying the data

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
