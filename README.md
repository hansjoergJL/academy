# Academy - Wissensdestillation für Spezialmodelle und LLM Finetuning

## Projektzusammenfassung

Entwicklung eines CLI-Tools zur Erstellung spezialisierter, kleiner Sprachmodelle (TinyLlama, lokal) durch Zuführung von Dokumenten (PDF), die mit Hilfe eines großen, leistungsfähigen Modells trainiert werden (wiederholte Datenzuführung in einfachen Sätzen: Datendestillation aus der Quelle).

## Hauptkomponenten

1. **Dokumenten-Einlesen:** Text aus verschiedenen Dokumentformaten extrahieren, insbesondere PDF
2. **Text-Zerlegung:** Dokumente in sinnvolle Abschnitte (Chunks) aufteilen
3. **Wissensdestillation:** Große Modelle nutzen, um aus den Textabschnitten vielfältige Sätze zu generieren
4. **Datenaufbereitung:** Generierte Sätze für das Training des kleinen Modells formatieren
5. **Modelltraining:** Das kleine Modell (TinyLlama, lokal) mit den aufbereiteten Daten fine-tunen
6. **Abfrage-Interface:** Interaktive Abfrage des trainierten Modells

## Projektstruktur

```
academy/
├── README.md
├── requirements.txt
├── config.yaml.template
├── setup_environment.py
├── academy/
│   ├── __init__.py
│   ├── cli.py              # Haupt-CLI mit train/query Commands
│   ├── config_manager.py   # Konfigurationsverwaltung
│   ├── document_processor.py # PDF-Extraktion und Chunking
│   ├── knowledge_distiller.py # OpenRouter API-Integration
│   ├── model_trainer.py    # TinyLlama Fine-Tuning
│   └── query_engine.py     # Abfragen des trainierten Modells
├── models/                 # Gespeicherte Modelle
├── data/
│   ├── raw/               # Original-PDFs
│   ├── processed/         # Verarbeitete Chunks
│   └── training/          # Trainingsdaten
└── tests/
```

## Technologien

- **Python 3.8+**
- **pypdf** (PDF-Textextraktion)
- **OpenRouter API** (Zugriff auf große KI-Modelle)
- **Hugging Face Transformers** (TinyLlama Fine-Tuning mit LoRA)
- **Click** (CLI-Interface)
- **PyYAML** (Konfigurationsverwaltung)
- **Datasets** (Trainingsdaten-Management)

## Konfiguration

Kopieren Sie `config.yaml.template` nach `config.yaml` und passen Sie die Einstellungen an:

### Wichtige Konfigurationsparameter

- **OpenRouter API Zugangsdaten**
- **Modellnamen** (both für Destillation und Training)
- **HuggingFace Model Pfade**
- **Training Parameter**
- **Daten Pfade**
- **Chunking Einstellungen**

## CLI Commands

```bash
# Training mit neuen Dokumenten (bereitet Daten vor, Training manuell)
academy train --input document.pdf --model-name my-specialist

# Abfragen eines trainierten Modells
academy query --model-name my-specialist

# Auflisten verfügbarer Modelle
academy list-models

# Initialisierung des Projekts
academy init
```

### Manuelles Modell-Training

Falls das Training in der CLI noch nicht implementiert ist, kann es manuell durchgeführt werden:

```python
from academy.model_trainer import ModelTrainer
trainer = ModelTrainer()
trainer.train_model("data/training/[model_name]_training.json", "[model_name]")
```

Beispiel für das Fine-Tuning-Leitfaden Modell:
```python
trainer.train_model("data/training/fine_tuning_guide_training.json", "fine_tuning_guide")
```

## Technik

**Besonderheiten:**
- PDF-Textextraktion mit `pypdf`
- OpenRouter API-Integration mit kostenlosen Modellen
- TinyLlama Fine-Tuning mit Hugging Face Transformers und LoRA
- CLI-Struktur mit Konfigurationsdatei

**Anforderungen:**
- Mindestens 8GB RAM, 16GB+ empfohlen
- OpenRouter API Key
- Python 3.8+ Umgebung

## Umgesetzter Implementierungsplan

### Phase 1: Environment Setup
1. ✅ Projektstruktur erstellen
2. ⏳ Requirements.txt definieren
3. ⏳ Config Template erstellen
4. ⏳ Environment Setup Skript
5. ⏳ Grundlegende CLI Struktur

### Phase 2: Kernkomponenten
1. ⏳ Konfigurations-Manager
2. ⏳ Dokumenten-Prozessor (PDF + Chunking)
3. ⏳ OpenRouter Integration
4. ⏳ Basis CLI Commands

### Phase 3: ML Pipeline
1. ⏳ Wissens-Destillation
2. ⏳ TinyLlama Trainer
3. ⏳ Datenformatierung
4. ⏳ Modell-Speicherung

### Phase 4: Query & Integration
1. ⏳ Query Engine
2. ⏳ Modell-Management
3. ⏳ Error Handling
4. ⏳ Logging System

### Phase 5: Testing & Optimierung
1. ⏳ Unit Tests
2. ⏳ Integration Tests
3. ⏳ Performance Optimierung
4. ⏳ Dokumentation

## Explizite Vorgehensweise für Modell-Training

Um ein oder mehrere Dokumente als Basis für ein spezialisiertes Sprachmodell zu verwenden, das anschließend einfach befragt werden kann, folgen Sie dieser Schritt-für-Schritt-Anleitung:

### 1. Umgebung einrichten
```bash
# Abhängigkeiten installieren
pip install -r requirements.txt

# Environment Setup ausführen
python setup_environment.py
```

### 2. Konfiguration vorbereiten
```bash
# Konfigurationsdatei erstellen
cp config.yaml.template config.yaml

# config.yaml bearbeiten:
# - OPENROUTER_API_KEY setzen (Umgebungsvariable oder direkt in der Datei)
# - use_cpu: true setzen, falls keine GPU verfügbar
# - Andere Parameter anpassen (Modellnamen, Pfade, etc.)
```

### 3. Dokumente vorbereiten
```bash
# Dokumente in das data/raw Verzeichnis kopieren
cp dokument.pdf data/raw/

# Mehrere Dokumente sind möglich - alle PDFs im raw Verzeichnis werden verarbeitet
cp dokument2.pdf data/raw/
cp dokument3.pdf data/raw/
```

### 4. Wissensdestillation durchführen
```bash
# CLI verwenden (empfohlen)
academy train --input data/raw/dokument.pdf --model-name mein-modell

# Für mehrere Dokumente: Mehrere --input Parameter oder alle PDFs im raw Verzeichnis
academy train --input data/raw/ --model-name mein-modell
```

**Was passiert in diesem Schritt:**
- Dokumente werden in Text konvertiert
- Text wird in Chunks aufgeteilt
- Mit Hilfe der OpenRouter API werden aus jedem Chunk einfache Sätze generiert
- Die generierten Sätze werden als JSON-Trainingsdaten gespeichert

### 5. Modell fine-tunen
```bash
# Das Training startet automatisch nach der Destillation
# Bei Bedarf manuelles Training:
python -c "
from academy.config_manager import ConfigManager
from academy.model_trainer import ModelTrainer
config = ConfigManager()
trainer = ModelTrainer(config)
trainer.train_model('data/training/mein-modell_training.json', 'mein-modell')
"
```

**Was passiert beim Training:**
- TinyLlama-Modell wird mit LoRA-Technik fine-getuned
- Training basiert auf den destillierten Sätzen
- Modell wird in models/custom/mein-modell gespeichert

### 6. Modell abfragen
```bash
# Interaktive Abfrage starten
academy query --model-name mein-modell

# Einzelne Frage stellen
academy query --model-name mein-modell --prompt "Was ist der Hauptinhalt des Dokuments?"
```

### Vollständiges Beispiel
```bash
# 1. Setup
pip install -r requirements.txt
python setup_environment.py

# 2. Konfiguration
cp config.yaml.template config.yaml
# Bearbeiten Sie config.yaml mit Ihrem API-Key

# 3. Dokumente vorbereiten
cp mein-dokument.pdf data/raw/

# 4. Training durchführen
academy train --input data/raw/mein-dokument.pdf --model-name spezialist-modell

# 5. Abfragen
academy query --model-name spezialist-modell
# Beispiel-Frage: "Erkläre die wichtigsten Konzepte aus dem Dokument"
```

### Hinweise
- **Zeitaufwand**: Destillation kann je nach Dokumentlänge 10-60 Minuten dauern
- **Hardware**: CPU-Training benötigt mindestens 8GB RAM, GPU wird empfohlen
- **Kosten**: OpenRouter API hat kostenlose Kontingente, aber Limits beachten
- **Qualität**: Die Qualität des Modells hängt von der Dokumentqualität ab

## Entwickler

Hans-Jörg Jödike, JL Software Solutions GbR.

## License

MIT
