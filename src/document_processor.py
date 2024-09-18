import json
import os
from typing import List
from llama_index.core import Document

def load_metadata(csv_folder):
    metadata_file = os.path.join(csv_folder, 'metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return {}

def enhance_documents_with_metadata(documents: List[Document], metadata):
    enhanced_documents = []
    for doc in documents:
        file_name = "AA.csv"
        if file_name in metadata:
            file_metadata = metadata[file_name]
            doc.metadata['file_description'] = file_metadata.get('file_description', '')
            
            columns_info = ""
            for col, info in file_metadata.get('columns', {}).items():
                columns_info += f"{col}: {info['description']} ({info['type']})\n"
            doc.metadata['columns_info'] = columns_info
            
            doc.text = f"File Description: {file_metadata.get('file_description', '')}\n\n" \
                       f"Columns Information:\n{columns_info}\n\n" \
                       f"Data:\n{doc.text}"
        
        enhanced_documents.append(doc)
    return enhanced_documents
