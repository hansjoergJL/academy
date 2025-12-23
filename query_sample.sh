#!/bin/bash

# Academy Query Script für Fine-Tuning Guide Modell
# Dieses Skript lädt das trainierte Modell und ermöglicht Abfragen.

MODEL_NAME="fine_tuning_guide"
MODEL_PATH="models/custom/$MODEL_NAME"

# Vorgeschlagene Frage
DEFAULT_QUESTION="Wie funktioniert der vollständige Prozess in Academy, um aus einem PDF-Dokument ein spezialisiertes TinyLlama-Modell zu erstellen? Beschreibe die Schritte von der Dokumentenverarbeitung über die Wissensdestillation bis zum Fine-Tuning."

echo "Academy Query Interface für Modell: $MODEL_NAME"
echo "Modell-Pfad: $MODEL_PATH"
echo ""

if [ ! -d "$MODEL_PATH" ]; then
    echo "Fehler: Modell nicht gefunden unter $MODEL_PATH"
    echo "Führen Sie zuerst train_sample.sh aus."
    exit 1
fi

echo "Stelle vorgeschlagene Frage:"
echo "$DEFAULT_QUESTION"
echo ""

# Führe Query mit der Academy CLI aus
python academy/cli.py query --model-name "$MODEL_NAME" --prompt "$DEFAULT_QUESTION"

# Prüfe, ob interaktiver Modus gewünscht ist
if [ "$1" = "-i" ]; then
    echo ""
    echo "Wechsle zu interaktivem Modus..."
    python academy/cli.py query --model-name "$MODEL_NAME" --interactive
else
    echo ""
    echo "Für interaktive Abfragen verwenden Sie: $0 -i"
fi