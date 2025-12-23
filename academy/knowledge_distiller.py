"""
Knowledge Distiller für Academy
Verwendet große Modelle zur Wissensextraktion und Datengenerierung
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, track

# Importiere Config Manager
sys.path.insert(0, str(Path(__file__).parent))
from config_manager import ConfigManager

console = Console()


@dataclass
class TrainingExample:
    """Repräsentiert ein Trainingsbeispiel für das kleine Modell"""
    input_text: str
    output_text: str
    metadata: Dict[str, Any]
    example_type: str  # 'knowledge_extraction', 'qa_pair', 'summarization'


class KnowledgeDistiller:
    """Wissensdestillation mit großen Sprachmodellen"""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialisiere KnowledgeDistiller"""
        self.config = config_manager
        self.openrouter_config = config_manager.get_openrouter_config()
        self.distillation_config = config_manager.get('distillation', {})
        
        # API Konfiguration
        self.api_key = self.openrouter_config['api_key']
        self.model_name = self.openrouter_config['model_name']
        self.base_url = self.openrouter_config['base_url']
        self.timeout = self.openrouter_config['timeout']
        self.max_retries = self.openrouter_config['max_retries']
        self.rate_limit_delay = self.openrouter_config['rate_limit_delay']
        
        # Prompt Templates
        self.prompts = self.distillation_config.get('prompts', {})
        self.examples_per_chunk = self.distillation_config.get('examples_per_chunk', 5)
        
        # Qualitätssicherung
        self.min_sentence_length = self.distillation_config.get('min_sentence_length', 10)
        self.max_sentence_length = self.distillation_config.get('max_sentence_length', 200)
        self.filter_duplicates = self.distillation_config.get('filter_duplicates', True)
        
        # Debug output
        console.print(f"[yellow]ℹ️[/yellow] Debug: target_total_sentences = {self.distillation_config.get('target_total_sentences', 'NOT_FOUND')}")
        
        # Initialisiere API Client
        self._init_api_client()
    
    def _init_api_client(self):
        """Initialisiere OpenRouter API Client"""
        try:
            from openai import OpenAI
            
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            console.print("[green]✅[/green] OpenRouter Client initialisiert")
        except ImportError:
            raise ImportError("OpenAI library nicht verfügbar. Installieren mit: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Fehler bei API Initialisierung: {e}")
    
    def test_connection(self) -> bool:
        """Teste Verbindung zur API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10
            )
            console.print("[green]✅[/green] API Verbindung erfolgreich")
            return True
        except Exception as e:
            console.print(f"[red]❌[/red] API Verbindung fehlgeschlagen: {e}")
            return False
    
    def distill_chunks(self, chunks: List, output_path: Optional[Union[str, Path]] = None) -> List[TrainingExample]:
        """Führe Wissensdestillation auf Chunks durch"""
        target_total = self.distillation_config.get('target_total_sentences', 10000)
        console.print(f"[blue]🧠[/blue] Generiere {target_total} einfache Sätze aus {len(chunks)} Chunks")
        
        all_examples = []
        
        # Metadaten für Satzberechnung vorbereiten
        for chunk in chunks:
            if hasattr(chunk, 'metadata'):
                chunk.metadata['total_chunks'] = len(chunks)
            else:
                chunk['metadata']['total_chunks'] = len(chunks)
        
        with Progress() as progress:
            task = progress.add_task("Generiere Sätze...", total=len(chunks))
            
            for i, chunk in enumerate(track(chunks, "Generiere Sätze...")):
                try:
                    examples = self._process_chunk(chunk)
                    all_examples.extend(examples)
                    
                    # Rate Limiting
                    if self.rate_limit_delay > 0:
                        time.sleep(self.rate_limit_delay)
                    
                    # Update progress
                    progress.update(task, advance=1, description=f"Generiere Sätze... {len(all_examples)}/{target_total}")
                    
                except Exception as e:
                    console.print(f"[red]❌[/red] Fehler bei Chunk {chunk.chunk_id}: {e}")
                    continue
        
        console.print(f"[green]✅[/green] {len(all_examples)} Sätze generiert (Ziel: {target_total})")
        
        # Speichere Ergebnisse
        if output_path:
            self._save_examples(all_examples, output_path)
        
        return all_examples
    
    def _process_chunk(self, chunk) -> List[TrainingExample]:
        """Verarbeite einen einzelnen Chunk"""
        examples = []
        
        # Extrahiere Text aus Chunk
        if hasattr(chunk, 'text'):
            text = chunk.text
            metadata = chunk.metadata
        else:
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
        
        # Nur einfache Sätze generieren gemäß Spezifikation
        try:
            knowledge_examples = self._generate_knowledge_extraction(text, metadata)
            examples.extend(knowledge_examples)
            
        except Exception as e:
            console.print(f"[yellow]⚠️[/yellow] Fehler bei Verarbeitung von {chunk.chunk_id}: {e}")
            return []
        
        # Qualitätssicherung
        examples = self._filter_and_validate_examples(examples)
        
        return examples
    
    def _generate_knowledge_extraction(self, text: str, metadata: Dict[str, Any]) -> List[TrainingExample]:
        """Generiere einfache Sätze für Trainingsdaten"""
        prompt_template = self.prompts.get('knowledge_extraction', 
            "Extrahiere aus dem folgenden Text genau {target_sentences} einfache, klare Sätze. Jeder Satz muss eine eigenständige, vollständige Information enthalten und einfach verständlich sein. Formuliere als Aussagesatz.\n\nText: {chunk}\n\n{target_sentences} Sätze:")
        
        # Berechne benötigte Sätze pro Chunk basierend auf Zielanzahl
        total_chunks = metadata.get('total_chunks', 1)
        target_total = self.distillation_config.get('target_total_sentences', 10000)
        sentences_per_chunk = max(1, target_total // total_chunks)
        
        prompt = prompt_template.format(chunk=text, target_sentences=sentences_per_chunk)
        
        try:
            response = self._call_api(prompt, max_tokens=800)
            examples = []
            
            # Teile Antwort in einzelne Sätze
            sentences = self._split_into_sentences(response)
            
            for sentence in sentences:
                if self._is_valid_sentence(sentence):
                    example = TrainingExample(
                        input_text="",  # Einfache Sätze benötigen keinen Input
                        output_text=sentence.strip(),
                        metadata=metadata.copy(),
                        example_type='simple_sentence'
                    )
                    examples.append(example)
            
            return examples
            
        except Exception as e:
            console.print(f"[yellow]⚠️[/yellow] Satzgenerierung fehlgeschlagen: {e}")
            return []
    
    def _generate_qa_pairs(self, text: str, metadata: Dict[str, Any]) -> List[TrainingExample]:
        """Generiere Frage-Antwort Paare"""
        prompt_template = self.prompts.get('qa_generation',
            "Erstelle basierend auf dem folgenden Text 3 Frage-Antwort Paare. Die Fragen sollten unterschiedlich schwierig sein.\n\nText: {chunk}\n\nFrage-Antwort Paare:")
        
        prompt = prompt_template.format(chunk=text)
        
        try:
            response = self._call_api(prompt)
            examples = []
            
            # Extrahiere QA Paare
            qa_pairs = self._extract_qa_pairs(response)
            
            for question, answer in qa_pairs:
                if self._is_valid_sentence(question) and self._is_valid_sentence(answer):
                    example = TrainingExample(
                        input_text=f"Frage: {question}",
                        output_text=f"Antwort: {answer}",
                        metadata=metadata.copy(),
                        example_type='qa_pair'
                    )
                    examples.append(example)
            
            return examples
            
        except Exception as e:
            console.print(f"[yellow]⚠️[/yellow] QA Generierung fehlgeschlagen: {e}")
            return []
    
    def _generate_summaries(self, text: str, metadata: Dict[str, Any]) -> List[TrainingExample]:
        """Generiere Zusammenfassungen"""
        prompt_template = self.prompts.get('summarization',
            "Fasse den folgenden Text in 2-3 prägnanten Sätzen zusammen.\n\nText: {chunk}\n\nZusammenfassung:")
        
        prompt = prompt_template.format(chunk=text)
        
        try:
            response = self._call_api(prompt)
            examples = []
            
            # Teile Antwort in Sätze
            sentences = self._split_into_sentences(response)
            
            for sentence in sentences:
                if self._is_valid_sentence(sentence):
                    example = TrainingExample(
                        input_text=text,
                        output_text=sentence.strip(),
                        metadata=metadata.copy(),
                        example_type='summarization'
                    )
                    examples.append(example)
            
            return examples
            
        except Exception as e:
            console.print(f"[yellow]⚠️[/yellow] Zusammenfassung fehlgeschlagen: {e}")
            return []
    
    def _call_api(self, prompt: str, max_tokens: int = 500) -> str:
        """Rufe OpenRouter API auf"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                
                content = response.choices[0].message.content
                if content:
                    return content.strip()
                else:
                    raise Exception("API returned empty content")
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    console.print(f"[yellow]⚠️[/yellow] API Fehler, versuche erneut in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e
        
        # Should never reach here
        return ""
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Teile Text in Sätze"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_qa_pairs(self, response: str) -> List[tuple]:
        """Extrahiere Frage-Antwort Paare aus Antwort"""
        qa_pairs = []
        
        # Versuche verschiedene Formate zu erkennen
        import re
        
        # Format 1: "Frage: ... Antwort: ..."
        pattern1 = r'Frage:\s*(.+?)\s*Antwort:\s*(.+?)(?=Frage:|$)'
        matches = re.findall(pattern1, response, re.IGNORECASE | re.DOTALL)
        for question, answer in matches:
            qa_pairs.append((question.strip(), answer.strip()))
        
        # Format 2: "Q: ... A: ..."
        pattern2 = r'Q:\s*(.+?)\s*A:\s*(.+?)(?=Q:|$)'
        matches = re.findall(pattern2, response, re.IGNORECASE | re.DOTALL)
        for question, answer in matches:
            qa_pairs.append((question.strip(), answer.strip()))
        
        # Format 3: Nummerierte Liste
        pattern3 = r'\d+\.\s*(.+?)\s*-\s*(.+?)(?=\n|\r|$)'
        matches = re.findall(pattern3, response)
        for question, answer in matches:
            qa_pairs.append((question.strip(), answer.strip()))
        
        return qa_pairs
    
    def _is_valid_sentence(self, sentence: str) -> bool:
        """Prüfe ob Satz qualitativ gültig ist"""
        sentence = sentence.strip()
        
        # Mindestlänge
        if len(sentence) < self.min_sentence_length:
            return False
        
        # Maximallänge
        if len(sentence) > self.max_sentence_length:
            return False
        
        # Muss mindestens ein Verb enthalten (einfache Prüfung)
        if not any(word in sentence.lower() for word in ['ist', 'sind', 'hat', 'haben', 'wird', 'können', 'ist', 'war']):
            return len(sentence) > 50  # Längere Sätze erlauben
        
        return True
    
    def _filter_and_validate_examples(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """Filtere und validiere Trainingsbeispiele"""
        if not self.filter_duplicates:
            return examples
        
        # Entferne Duplikate basierend auf output_text
        seen_outputs = set()
        filtered_examples = []
        
        for example in examples:
            if example.output_text not in seen_outputs:
                seen_outputs.add(example.output_text)
                filtered_examples.append(example)
        
        return filtered_examples
    
    def _save_examples(self, examples: List[TrainingExample], output_path: Union[str, Path]) -> None:
        """Speichere Trainingsbeispiele"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Konvertiere zu JSON-Format
        examples_data = []
        for example in examples:
            example_data = {
                'input_text': example.input_text,
                'output_text': example.output_text,
                'metadata': example.metadata,
                'example_type': example.example_type
            }
            examples_data.append(example_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(examples_data, f, ensure_ascii=False, indent=2)
        
        console.print(f"[green]✅[/green] {len(examples)} Trainingsbeispiele gespeichert: {output_path}")
    
    def load_examples(self, input_path: Union[str, Path]) -> List[TrainingExample]:
        """Lade Trainingsbeispiele aus Datei"""
        input_path = Path(input_path)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            examples_data = json.load(f)
        
        examples = []
        for example_data in examples_data:
            example = TrainingExample(
                input_text=example_data['input_text'],
                output_text=example_data['output_text'],
                metadata=example_data['metadata'],
                example_type=example_data['example_type']
            )
            examples.append(example)
        
        return examples
    
    def format_for_training(self, examples: List[TrainingExample], output_path: Union[str, Path]) -> None:
        """Formattiere Beispiele für TinyLlama Training"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        training_data = []
        
        for example in examples:
            # Für einfache Sätze: Nur Text-To-Text Format
            if example.example_type == 'simple_sentence':
                training_item = {
                    "text": example.output_text
                }
            else:
                # Format für Chat-Finetuning
                messages = [
                    {
                        "role": "user",
                        "content": example.input_text
                    },
                    {
                        "role": "assistant", 
                        "content": example.output_text
                    }
                ]
                training_item = {
                    "messages": messages
                }
            
            training_data.append(training_item)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        console.print(f"[green]✅[/green] {len(training_data)} Trainingsdaten formatiert: {output_path}")


def main():
    """Test-Funktion für KnowledgeDistiller"""
    import argparse
    
    parser = argparse.ArgumentParser(description='KnowledgeDistiller Test')
    parser.add_argument('--input', '-i', required=True, help='Chunks JSON Datei')
    parser.add_argument('--output', '-o', required=True, help='Ausgabedatei für Trainingsbeispiele')
    parser.add_argument('--config', '-c', help='Konfigurationsdatei')
    parser.add_argument('--test', action='store_true', help='Nur API Verbindung testen')
    parser.add_argument('--dry-run', action='store_true', help='Nur Konfiguration testen')
    
    args = parser.parse_args()
    
    # Initialisiere Config Manager
    config_manager = ConfigManager(args.config)
    
    # Initialisiere KnowledgeDistiller
    distiller = KnowledgeDistiller(config_manager)
    
    if args.test:
        # Teste API Verbindung
        distiller.test_connection()
        return
    
    if args.dry_run:
        # Teste nur Konfiguration
        console.print("[green]✅[/green] Konfiguration validiert")
        return
    
    # Lade Chunks
    from document_processor import DocumentProcessor
    processor = DocumentProcessor(config_manager)
    
    chunks = processor.load_chunks(args.input)
    console.print(f"[green]✅[/green] {len(chunks)} Chunks geladen")
    
    # Führe Destillation durch
    examples = distiller.distill_chunks(chunks, args.output)
    
    # Formattiere für Training
    training_path = Path(args.output).with_suffix('.training.json')
    distiller.format_for_training(examples, training_path)


if __name__ == "__main__":
    main()