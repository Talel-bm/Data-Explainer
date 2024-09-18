import csv
from pathlib import Path
from typing import List
from datetime import datetime
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

class ImprovedCSVReader(BaseReader):
    def load_data(self, file: Path) -> List[Document]:
        documents = []
        with open(file, 'r', newline='', encoding='latin-1') as f:
            reader = csv.reader(f, delimiter=';') 
            headers = next(reader, None)
            for i, row in enumerate(reader, start=2):
                try:
                    content = {}
                    for j, cell in enumerate(row):
                        if j < len(headers):
                            header = headers[j]
                            if header.startswith("DATE_"):
                                try:
                                    if ':' in cell:  # For DATE_EXECUTION format
                                        date_obj = datetime.strptime(cell, '%d%b%y:%H:%M:%S')
                                        cell = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                                    else:  # For other date formats
                                        date_obj = datetime.strptime(cell, '%d/%m/%Y')
                                        cell = date_obj.strftime('%Y-%m-%d')
                                except ValueError:
                                    pass
                            content[header] = cell

                    content_str = ", ".join(f"{k}: {v}" for k, v in content.items())
                    documents.append(Document(text=content_str))
                except Exception as e:
                    print(f"Error processing row {i} in {file}: {e}")
        return documents

def load_csv_directory(directory: str) -> List[Document]:
    reader = ImprovedCSVReader()
    documents = []
    for file in Path(directory).glob('*.csv'):
        documents.extend(reader.load_data(file))
    return documents
