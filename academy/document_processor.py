"""
Dokumentenprozessor für Academy
Extrahiert Text aus PDFs und anderen Dokumentformaten
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import re
from dataclasses import dataclass
from rich.console import Console

# Importiere Config Manager
sys.path.insert(0, str(Path(__file__).parent))
from config_manager import ConfigManager

console = Console()


@dataclass
class DocumentChunk:
    """Repräsentiert ein Text-Chunk aus einem Dokument"""
    text: str
    metadata: Dict[str, Any]
    source: Path
    chunk_id: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None


class DocumentProcessor:
    """Verarbeitung von Dokumenten für Academy"""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialisiere Dokumentenprozessor"""
        self.config = config_manager
        self.paths = config_manager.get_paths_config()
        self.docs_config = config_manager.get_documents_config()
        
        # Unterstützte Formate
        self.supported_formats = self.docs_config.get('supported_formats', ['.pdf', '.txt', '.md'])
        
        # Chunking Konfiguration
        self.chunking_config = self.docs_config.get('chunking', {})
        self.chunk_size = self.chunking_config.get('chunk_size', 1000)
        self.chunk_overlap = self.chunking_config.get('chunk_overlap', 200)
        self.strategy = self.chunking_config.get('strategy', 'recursive')
        self.separators = self.chunking_config.get('separators', ['\n\n', '\n', '. ', '? ', '! ', ' ', ''])
    
    def is_supported_format(self, file_path: Path) -> bool:
        """Prüfe ob Dateiformat unterstützt wird"""
        return file_path.suffix.lower() in self.supported_formats
    
    def process_directory(self, directory: Union[str, Path]) -> List[DocumentChunk]:
        """Verarbeite alle unterstützten Dokumente in einem Verzeichnis"""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Verzeichnis nicht gefunden: {directory}")
        
        all_chunks = []
        supported_files = []
        
        # Finde alle unterstützten Dateien
        for file_path in directory.rglob('*'):
            if file_path.is_file() and self.is_supported_format(file_path):
                supported_files.append(file_path)
        
        console.print(f"[green]✓[/green] {len(supported_files)} unterstützte Dokumente gefunden")
        
        # Verarbeite jede Datei
        for file_path in supported_files:
            try:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
                console.print(f"[green]✓[/green] Verarbeitet: {file_path.name} ({len(chunks)} Chunks)")
            except Exception as e:
                console.print(f"[red]❌[/red] Fehler bei {file_path.name}: {e}")
        
        console.print(f"[green]✅[/green] Insgesamt {len(all_chunks)} Chunks extrahiert")
        return all_chunks
    
    def process_file(self, file_path: Union[str, Path]) -> List[DocumentChunk]:
        """Verarbeite eine einzelne Datei"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
        
        if not self.is_supported_format(file_path):
            raise ValueError(f"Format nicht unterstützt: {file_path.suffix}")
        
        # Extrahiere Text basierend auf Dateityp
        if file_path.suffix.lower() == '.pdf':
            text, metadata = self._extract_pdf_text(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            text, metadata = self._extract_text_file(file_path)
        else:
            raise ValueError(f"Nicht unterstützter Dateityp: {file_path.suffix}")
        
        # Füge Metadaten hinzu
        metadata.update({
            'filename': file_path.name,
            'filepath': str(file_path),
            'file_size': file_path.stat().st_size,
            'extraction_method': self._get_extraction_method(file_path.suffix)
        })
        
        # Teile Text in Chunks
        chunks = self._chunk_text(text, metadata, file_path)
        
        return chunks
    
    def _extract_pdf_text(self, pdf_path: Path) -> tuple[str, Dict[str, Any]]:
        """Extrahiere Text aus PDF"""
        try:
            # Versuche zuerst PyPDF
            return self._extract_pdf_with_pypdf(pdf_path)
        except ImportError:
            console.print("[yellow]⚠️ PyPDF nicht verfügbar, versuche Alternative...[/yellow]")
            try:
                return self._extract_pdf_with_pymupdf(pdf_path)
            except ImportError:
                raise ImportError("Weder PyPDF noch PyMuPDF verfügbar. Bitte installieren Sie: pip install pypdf pymupdf")
    
    def _extract_pdf_with_pypdf(self, pdf_path: Path) -> tuple[str, Dict[str, Any]]:
        """Extrahiere Text mit PyPDF"""
        from pypdf import PdfReader
        
        reader = PdfReader(str(pdf_path))
        text_parts = []
        metadata = {
            'num_pages': len(reader.pages),
            'extraction_method': 'pypdf'
        }
        
        # Extrahiere Metadaten
        if reader.metadata:
            metadata.update({
                'title': reader.metadata.get('/Title', ''),
                'author': reader.metadata.get('/Author', ''),
                'subject': reader.metadata.get('/Subject', ''),
                'creator': reader.metadata.get('/Creator', '')
            })
        
        # Extrahiere Text von jeder Seite
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(f"=== Seite {page_num + 1} ===\n{page_text}")
            except Exception as e:
                console.print(f"[yellow]⚠️ Seite {page_num + 1} nicht lesbar: {e}[/yellow]")
                continue
        
        return '\n\n'.join(text_parts), metadata
    
    def _extract_pdf_with_pymupdf(self, pdf_path: Path) -> tuple[str, Dict[str, Any]]:
        """Extrahiere Text mit PyMuPDF"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF nicht verfügbar. Installieren mit: pip install pymupdf")
        
        doc = fitz.open(str(pdf_path))
        text_parts = []
        metadata = {
            'num_pages': len(doc),
            'extraction_method': 'pymupdf'
        }
        
        # Extrahiere Metadaten
        if doc.metadata:
            metadata.update({
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', '')
            })
        
        # Extrahiere Text von jeder Seite
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(f"=== Seite {page_num + 1} ===\n{page_text}")
        
        doc.close()
        return '\n\n'.join(text_parts), metadata
    
    def _extract_text_file(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Extrahiere Text aus Textdateien (.txt, .md)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Versuche andere Encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Konnte Datei nicht lesen: {file_path}")
        
        metadata = {
            'extraction_method': 'direct',
            'encoding': 'utf-8'
        }
        
        return text, metadata
    
    def _get_extraction_method(self, suffix: str) -> str:
        """Hole Extraktionsmethode für Dateityp"""
        if suffix.lower() == '.pdf':
            return 'pdf'
        elif suffix.lower() in ['.txt', '.md']:
            return 'text'
        else:
            return 'unknown'
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any], source: Path) -> List[DocumentChunk]:
        """Teile Text in Chunks basierend auf konfigurierter Strategie"""
        if self.strategy == 'recursive':
            return self._recursive_chunking(text, metadata, source)
        elif self.strategy == 'fixed':
            return self._fixed_chunking(text, metadata, source)
        elif self.strategy == 'semantic':
            return self._semantic_chunking(text, metadata, source)
        else:
            raise ValueError(f"Unbekannte Chunking-Strategie: {self.strategy}")
    
    def _recursive_chunking(self, text: str, metadata: Dict[str, Any], source: Path) -> List[DocumentChunk]:
        """Rekursives Chunking mit verschiedenen Trennzeichen"""
        chunks = []
        
        # Finde beste Trennzeichen
        for separator in self.separators:
            if separator and separator in text:
                sections = text.split(separator)
                if len(sections) > 1:
                    break
        else:
            # Fallback auf Fixed-Sized Chunking
            return self._fixed_chunking(text, metadata, source)
        
        # Erstelle Chunks aus Abschnitten
        current_chunk = ""
        chunk_id = 0
        
        for section in sections:
            test_chunk = current_chunk + separator + section if current_chunk else section
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Speichere aktuellen Chunk
                if current_chunk.strip():
                    chunks.append(DocumentChunk(
                        text=current_chunk.strip(),
                        metadata=metadata.copy(),
                        source=source,
                        chunk_id=f"{source.stem}_{chunk_id}"
                    ))
                    chunk_id += 1
                
                # Beginne neuen Chunk
                current_chunk = section
        
        # Speichere letzten Chunk
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                text=current_chunk.strip(),
                metadata=metadata.copy(),
                source=source,
                chunk_id=f"{source.stem}_{chunk_id}"
            ))
        
        # Füge Overlap hinzu
        if self.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _fixed_chunking(self, text: str, metadata: Dict[str, Any], source: Path) -> List[DocumentChunk]:
        """Fixed-Size Chunking"""
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i:i + self.chunk_size]
            
            chunks.append(DocumentChunk(
                text=chunk_text,
                metadata=metadata.copy(),
                source=source,
                chunk_id=f"{source.stem}_{chunk_id}"
            ))
            chunk_id += 1
        
        return chunks
    
    def _semantic_chunking(self, text: str, metadata: Dict[str, Any], source: Path) -> List[DocumentChunk]:
        """Semantisches Chunking (basiert auf Sätzen und Absätzen)"""
        # Teile in Sätze
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Speichere aktuellen Chunk
                if current_chunk.strip():
                    chunks.append(DocumentChunk(
                        text=current_chunk.strip(),
                        metadata=metadata.copy(),
                        source=source,
                        chunk_id=f"{source.stem}_{chunk_id}"
                    ))
                    chunk_id += 1
                
                # Beginne neuen Chunk
                current_chunk = sentence
        
        # Speichere letzten Chunk
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                text=current_chunk.strip(),
                metadata=metadata.copy(),
                source=source,
                chunk_id=f"{source.stem}_{chunk_id}"
            ))
        
        return chunks
    
    def _add_overlap(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Füge Overlap zwischen Chunks hinzu"""
        if self.chunk_overlap <= 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            text = chunk.text
            
            # Füge Text vom vorherigen Chunk hinzu
            if i > 0:
                prev_chunk = chunks[i-1].text
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                text = overlap_text + " " + text
            
            overlapped_chunks.append(DocumentChunk(
                text=text,
                metadata=chunk.metadata,
                source=chunk.source,
                chunk_id=chunk.chunk_id
            ))
        
        return overlapped_chunks
    
    def save_chunks(self, chunks: List[DocumentChunk], output_path: Union[str, Path]) -> None:
        """Speichere Chunks in Datei"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Speichere als JSON
        import json
        
        chunks_data = []
        for chunk in chunks:
            chunk_data = {
                'text': chunk.text,
                'metadata': chunk.metadata,
                'source': str(chunk.source),
                'chunk_id': chunk.chunk_id,
                'start_page': chunk.start_page,
                'end_page': chunk.end_page
            }
            chunks_data.append(chunk_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        console.print(f"[green]✅[/green] {len(chunks)} Chunks gespeichert: {output_path}")
    
    def load_chunks(self, input_path: Union[str, Path]) -> List[DocumentChunk]:
        """Lade Chunks aus Datei"""
        input_path = Path(input_path)
        
        import json
        
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        chunks = []
        for chunk_data in chunks_data:
            chunk = DocumentChunk(
                text=chunk_data['text'],
                metadata=chunk_data['metadata'],
                source=Path(chunk_data['source']),
                chunk_id=chunk_data['chunk_id'],
                start_page=chunk_data.get('start_page'),
                end_page=chunk_data.get('end_page')
            )
            chunks.append(chunk)
        
        return chunks


def main():
    """Test-Funktion für DocumentProcessor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DocumentProcessor Test')
    parser.add_argument('--input', '-i', required=True, help='Eingabedatei oder -verzeichnis')
    parser.add_argument('--output', '-o', help='Ausgabedatei für Chunks')
    parser.add_argument('--config', '-c', help='Konfigurationsdatei')
    
    args = parser.parse_args()
    
    # Initialisiere Config Manager
    config_manager = ConfigManager(args.config)
    
    # Initialisiere DocumentProcessor
    processor = DocumentProcessor(config_manager)
    
    # Verarbeite Input
    input_path = Path(args.input)
    
    if input_path.is_file():
        chunks = processor.process_file(input_path)
        console.print(f"[green]✅[/green] {len(chunks)} Chunks aus {input_path.name} extrahiert")
    elif input_path.is_dir():
        chunks = processor.process_directory(input_path)
    else:
        console.print(f"[red]❌[/red] Input nicht gefunden: {input_path}")
        return
    
    # Speichere Chunks
    if args.output:
        processor.save_chunks(chunks, args.output)
    else:
        # Zeige erste paar Chunks
        for i, chunk in enumerate(chunks[:3]):
            console.print(f"\n[bold]Chunk {i+1}:[/bold] {chunk.chunk_id}")
            console.print(f"[dim]Quelle: {chunk.source.name}[/dim]")
            console.print(chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text)


if __name__ == "__main__":
    main()