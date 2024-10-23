import os
import json
from PyPDF2 import PdfReader

def extract_pdf_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Erreur lors du traitement du PDF {pdf_path}: {str(e)}")
        return ""

def extract_txt_text(txt_path):
    # Try encodings commonly used for French text files
    encodings = ['cp1252', 'iso-8859-1', 'utf-8', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(txt_path, 'r', encoding=encoding) as file:
                content = file.read()
                # Verify if the content contains common French characters to validate encoding
                if any(char in content for char in 'éèêëàâäôöûüçîï'):
                    return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Erreur de lecture du fichier {txt_path}: {str(e)}")
            continue
    
    # If no encoding worked with French character validation, try one last time with cp1252
    try:
        with open(txt_path, 'r', encoding='cp1252') as file:
            return file.read()
    except Exception as e:
        print(f"Impossible de décoder le fichier {txt_path}: {str(e)}")
        return ""

def folder_to_jsonl(folder_path, output_jsonl_path):
    processed_files = 0
    failed_files = 0
    
    # Open the JSONL file in write mode
    with open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file:
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            try:
                if filename.lower().endswith('.pdf'):
                    text = extract_pdf_text(file_path)
                    file_type = "pdf"
                elif filename.lower().endswith('.txt'):
                    text = extract_txt_text(file_path)
                    file_type = "txt"
                else:
                    continue
                
                if text:  # Only write to file if text was successfully extracted
                    # Create a dictionary for this file
                    file_info = {
                        "file_name": filename,
                        "text": text
                    }
                    # Write the JSON object as a single line
                    jsonl_file.write(json.dumps(file_info, ensure_ascii=False) + '\n')
                    processed_files += 1
                    print(f"Traitement réussi: {filename}")
                else:
                    failed_files += 1
                    print(f"Échec du traitement: {filename}")
                    
            except Exception as e:
                failed_files += 1
                print(f"Erreur lors du traitement du fichier {filename}: {str(e)}")
                continue
    
    print(f"\nFichier JSONL créé: {output_jsonl_path}")
    print(f"Fichiers traités avec succès: {processed_files}")
    print(f"Fichiers échoués: {failed_files}")

# Example usage
folder_path = "---/general knowledge/pdf/to text"
output_jsonl_path = "training_text.jsonl"  # Changed extension to .jsonl
folder_to_jsonl(folder_path, output_jsonl_path)
