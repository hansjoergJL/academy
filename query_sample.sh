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

# Placeholder für Query Engine (noch nicht implementiert)
echo "Antwort: [Placeholder - Query Engine ist noch nicht implementiert]"
echo "Die Antwort würde hier vom Modell generiert werden."
echo ""

# Interaktive Abfrage-Schleife
while true; do
    echo "Geben Sie eine weitere Frage ein (oder 'exit' zum Beenden):"
    read -r question

    if [ "$question" = "exit" ]; then
        echo "Abfrage beendet."
        break
    fi

    echo "Frage: $question"
    echo "Antwort: [Placeholder - Implementierung ausstehend]"
    echo ""
done