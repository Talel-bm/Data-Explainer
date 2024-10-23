import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import sent_tokenize
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InstructionDatasetGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """Initialize the generator with the specified model."""
        self.model_name = model_name
        self.setup_model()
        
    def setup_model(self):
        """Set up the model and tokenizer with CPU optimization."""
        try:
            # Load in 8-bit to reduce memory usage
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",
                load_in_8bit=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            nltk.download('punkt')
            logger.info("Model and tokenizer initialized successfully on CPU")
        except Exception as e:
            logger.error(f"Error setting up model: {str(e)}")
            raise

    def read_jsonl(self, file_path: str) -> List[Dict]:
        """Read the input JSONL file."""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            logger.info(f"Successfully read {len(data)} documents from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error reading JSONL file: {str(e)}")
            raise

    def split_into_coherent_sections(self, text: str) -> List[str]:
        """Split text into coherent sections using sentence tokenization and grouping."""
        try:
            sentences = sent_tokenize(text, language='french')
            
            # Smaller sections for CPU processing (200 words instead of 300)
            sections = []
            current_section = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence.split())
                if current_length + sentence_length > 200 and current_section:
                    sections.append(' '.join(current_section))
                    current_section = [sentence]
                    current_length = sentence_length
                else:
                    current_section.append(sentence)
                    current_length += sentence_length
            
            if current_section:
                sections.append(' '.join(current_section))
                
            return sections
        except Exception as e:
            logger.error(f"Error splitting text into sections: {str(e)}")
            raise

    def generate_qa_pair(self, section: str) -> Tuple[str, str]:
        """Generate a question-answer pair from a given section."""
        try:
            # Enhanced prompt for Tunisian insurance law context
            messages = [
                {"role": "system", "content": """Vous êtes un expert spécialisé en droit et réglementation des assurances en Tunisie. 
Votre tâche est de générer des questions perspicaces et des réponses détaillées basées sur les sections du texte juridique fourni.

Lignes directrices pour la génération :
1. Concentrez-vous sur les concepts juridiques clés, les exigences réglementaires et les implications pratiques.
2. Tenez compte du contexte spécifique du marché tunisien de l'assurance.
3. Mettez en évidence les définitions légales importantes, les obligations et les exigences de conformité.
4. Incluez des références pertinentes aux codes et réglementations des assurances en Tunisie lorsque cela est applicable.
5. Assurez-vous que la réponse offre une compréhension complète tout en restant fidèle au texte source.

Générez votre réponse en français, en maintenant la précision et la clarté juridiques.
Formatez votre réponse exactement comme suit :
Question : [Votre question]
Réponse : [Votre réponse détaillée]
"""},
                {"role": "user", "content": section}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Optimize generation parameters for CPU
            model_inputs = self.tokenizer([text], return_tensors="pt")
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=2  # Reduced for CPU efficiency
            )
            
            response = self.tokenizer.batch_decode(
                [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)],
                skip_special_tokens=True
            )[0]
            
            try:
                question = response.split("Question:")[1].split("Answer:")[0].strip()
                answer = response.split("Answer:")[1].strip()
            except IndexError:
                logger.warning(f"Could not parse QA pair from response: {response}")
                question = "ERROR: Could not generate question"
                answer = "ERROR: Could not generate answer"
            
            return question, answer
        except Exception as e:
            logger.error(f"Error generating QA pair: {str(e)}")
            return "ERROR", str(e)

    def process_documents(self, input_file: str, output_file: str, batch_size: int = 5):
        """Process documents in small batches to manage memory."""
        try:
            documents = self.read_jsonl(input_file)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1}/{len(documents)//batch_size + 1}")
                    
                    for doc in batch:
                        logger.info(f"Processing document: {doc['file_name']}")
                        sections = self.split_into_coherent_sections(doc['messages'])
                        
                        for section in sections:
                            question, answer = self.generate_qa_pair(section)
                            
                            instruction_example = {
                                "instruction": question,
                                "input": "",
                                "output": answer,
                                "source_document": doc['file_name']
                            }
                            
                            f.write(json.dumps(instruction_example, ensure_ascii=False) + '\n')
                        
                        # Clear some memory
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
            logger.info(f"Successfully generated instruction dataset: {output_file}")
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

if __name__ == "__main__":
    generator = InstructionDatasetGenerator()
    generator.process_documents(
        input_file='trainingtext__.jsonl',
        output_file='instruction_qwen.jsonl'
    )
