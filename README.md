# Academy - Wissensdestillation für Spezialmodelle (LLM Finetuning)

**Entwickler:** Hans-Jörg Jödike, JL Software Solutions GbR.

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
# Training mit neuen Dokumenten
academy train --input document.pdf --model-name my-specialist

# Abfragen eines trainierten Modells
academy query --model-name my-specialist

# Auflisten verfügbarer Modelle
academy list-models

# Initialisierung des Projekts
academy init
```

## Umsetzbarkeits-Analyse

**Vollständig umsetzbar:**
- PDF-Textextraktion mit `pypdf`
- OpenRouter API-Integration mit kostenlosen Modellen
- TinyLlama Fine-Tuning mit Hugging Face Transformers und LoRA
- CLI-Struktur mit Konfigurationsdatei

**Anforderungen:**
- Mindestens 8GB RAM, 16GB+ empfohlen
- OpenRouter API Key
- Python 3.8+ Umgebung

## Schrittweise Implementierungsplan

### Phase 1: Environment Setup (Woche 1)
1. ✅ Projektstruktur erstellen
2. ⏳ Requirements.txt definieren
3. ⏳ Config Template erstellen
4. ⏳ Environment Setup Skript
5. ⏳ Grundlegende CLI Struktur

### Phase 2: Kernkomponenten (Woche 2-3)
1. ⏳ Konfigurations-Manager
2. ⏳ Dokumenten-Prozessor (PDF + Chunking)
3. ⏳ OpenRouter Integration
4. ⏳ Basis CLI Commands

### Phase 3: ML Pipeline (Woche 3-4)
1. ⏳ Wissens-Destillation
2. ⏳ TinyLlama Trainer
3. ⏳ Datenformatierung
4. ⏳ Modell-Speicherung

### Phase 4: Query & Integration (Woche 4-5)
1. ⏳ Query Engine
2. ⏳ Modell-Management
3. ⏳ Error Handling
4. ⏳ Logging System

### Phase 5: Testing & Optimierung (Woche 5-6)
1. ⏳ Unit Tests
2. ⏳ Integration Tests
3. ⏳ Performance Optimierung
4. ⏳ Dokumentation

## Geschätzter Entwicklungsaufwand

- **MVP (Minimum Viable Product)**: 2-3 Wochen
- **Vollständige Implementierung**: 4-6 Wochen
- **Testing & Optimierung**: 1-2 Wochen

## Kritische Erfolgsfaktoren

1. **OpenRouter API Key** für den Zugriff auf kostenlose Modelle
2. **Hardware-Anforderungen**: Mindestens 8GB RAM, 16GB+ empfohlen
3. **Dokumentqualität**: Gute OCR-Voraussetzungen für PDFs
4. **Chunking-Strategie**: Balance zwischen Kontext und Spezifität

## Nutzungsbeispiel

```bash
# 1. Environment einrichten
python setup_environment.py

# 2. Konfiguration anpassen
cp config.yaml.template config.yaml
# Edit config.yaml mit OpenRouter API Key

# 3. PDF verarbeiten und Modell trainieren
academy train --input fachbuch.pdf --model-name mathematik-experte

# 4. Abfragen des trainierten Modells
academy query --model-name mathematik-experte
```

## License

[License Type - hinzufügen]