# TECH.md - Technische Dokumentation für Academy

## Übersicht

Academy implementiert eine Pipeline für Wissensdestillation und Fine-Tuning von Sprachmodellen. Das System konvertiert Dokumente in spezialisierte, lokale Sprachmodelle durch automatisierte Datenvorbereitung und maschinelles Lernen.

## Architektur

### Kernkomponenten

1. **Dokumentenverarbeitung** (`document_processor.py`)
2. **Wissensdestillation** (`knowledge_distiller.py`)
3. **Modelltraining** (`model_trainer.py`)
4. **Konfigurationsmanagement** (`config_manager.py`)
5. **CLI-Interface** (`cli.py`)

## Technische Schritte im Detail

### 1. Dokumentenverarbeitung

**Funktion**: Extraktion von Text aus Dokumentformaten, insbesondere PDF-Dateien.

**Technologie**:
- **pypdf** (Version >=3.0.0): PDF-Parsing und Textextraktion
- Alternativ: **pymupdf** (MuPDF) für komplexere Layouts

**Algorithmus**:
```python
# PDF zu Text Konvertierung
with open(pdf_path, 'rb') as file:
    pdf_reader = pypdf.PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
```

**Chunking-Strategie**:
- **Recursive Text Splitter**: Text wird hierarchisch in Abschnitte aufgeteilt
- Separatoren: `\n\n`, `\n`, `. `, `? `, `! `, ` `
- Chunk-Größe: 1000 Zeichen mit 200 Zeichen Überlappung
- Strategie: "recursive" für kontextuelle Kohärenz

### 2. Wissensdestillation

**Funktion**: Generierung von Trainingsdaten aus Text-Chunks mittels Large Language Models (LLMs).

**Technologie**:
- **OpenRouter API**: Zugang zu verschiedenen LLM-Providern
- **openai** (Version >=1.0.0): Python-Client für OpenAI-kompatible APIs
- **requests** (Version >=2.31.0): HTTP-Kommunikation

**Algorithmus**:
```python
# Prompt-Template für Satzgenerierung
prompt = f"""
Extrahiere aus dem folgenden Text genau {target_sentences} einfache, klare Sätze.
Jeder Satz muss eine eigenständige, vollständige Information enthalten.

Text: {chunk}

{target_sentences} Sätze:
"""

# API-Aufruf an OpenRouter
response = openai.OpenAI(api_key=api_key, base_url=base_url).chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7
)
```

**Qualitätssicherung**:
- Satz-Längenfilter: 10-200 Zeichen
- Duplikatentfernung
- Validierung auf Aussagesatz-Struktur

### 3. Datenaufbereitung

**Funktion**: Formatierung der generierten Sätze für das Modelltraining.

**Technologie**:
- **pandas** (Version >=2.0.0): Datenmanipulation
- **datasets** (Version >=2.14.0): Hugging Face Dataset-Format

**Format**:
```json
{
  "text": "Der Satz ist eine eigenständige Information.",
  "source": "chunk_001",
  "document": "dokument.pdf"
}
```

### 4. Modell-Fine-Tuning

**Funktion**: Anpassung eines vortrainierten Sprachmodells an die spezifischen Trainingsdaten.

**Technologie**:
- **transformers** (Version >=4.35.0): Hugging Face Transformers Framework
- **peft** (Version >=0.7.0): Parameter-Efficient Fine-Tuning (LoRA)
- **torch** (Version >=2.0.0): PyTorch Deep Learning Framework
- **accelerate** (Version >=0.24.0): Training-Beschleunigung

**Algorithmus**:
```python
# LoRA-Konfiguration
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05
)

# Modell laden und PEFT anwenden
model = AutoModelForCausalLM.from_pretrained(base_model)
model = get_peft_model(model, lora_config)

# Training-Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)
```

**Training-Parameter**:
- Batch Size: 1 (für CPU-Kompatibilität)
- Learning Rate: 2e-4
- Epochen: 3
- Max Sequence Length: 512
- Optimierer: AdamW

### 5. Modell-Abfrage

**Funktion**: Interaktive Nutzung des fine-getunten Modells.

**Technologie**:
- **transformers**: Pipeline für Text-Generierung
- **torch**: Modell-Inferenz

**Algorithmus**:
```python
# Modell laden
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Pipeline erstellen
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,
    do_sample=True
)

# Abfrage generieren
response = pipeline(f"Kontext: {prompt}\nAntwort:")[0]['generated_text']
```

## Datenfluss

```
PDF-Dokumente
    ↓
Textextraktion (pypdf)
    ↓
Chunking (recursive splitter)
    ↓
Wissensdestillation (OpenRouter API + LLM)
    ↓
Datenformatierung (JSON)
    ↓
Fine-Tuning (TinyLlama + LoRA)
    ↓
Trained Model (lokale Dateien)
    ↓
Query Interface (transformers pipeline)
```

## Hardware-Anforderungen

- **CPU-Modus**: Mindestens 8GB RAM, 16GB+ empfohlen
- **GPU-Modus**: CUDA-kompatible GPU mit mindestens 4GB VRAM
- **Speicher**: 10-20GB freier Festplattenspeicher für Modelle und Daten

## Externe APIs

- **OpenRouter API**: Kostenloses Kontingent verfügbar, Rate-Limits beachten
- **Hugging Face Hub**: Optional für Modell-Upload (kostenlos)

## Fehlerbehandlung

- **API-Limits**: Automatische Retry-Logik mit exponentiellem Backoff
- **Modell-Fehler**: Fallback auf Standardkonfiguration
- **Speicherfehler**: Batch-Verarbeitung und Chunking-Optimierung

## Monitoring und Logging

- **loguru** (Version >=0.7.0): Strukturiertes Logging
- **tqdm** (Version >=4.65.0): Fortschrittsanzeigen
- **rich** (Version >=13.0.0): Terminal-UI für bessere Benutzererfahrung

## Sicherheit

- **API-Keys**: Environment-Variablen und .env-Dateien
- **Dateizugriffe**: Beschränkung auf konfigurierte Verzeichnisse
- **Modell-Validierung**: Integritätsprüfungen vor Training

## Performance-Optimierungen

- **LoRA**: Reduziert Trainingskosten um 90%+ gegenüber Full Fine-Tuning
- **Quantisierung**: Optionale 8-bit Training für geringeren Speicherbedarf
- **Batch-Processing**: Parallele Verarbeitung für Destillation