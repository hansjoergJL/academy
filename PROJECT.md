# Projektzusammenfassung: Wissensdestillation für Spezialmodelle (LLM Finetuning)

**Projektname:** Academy

**Entwickler:** Hans-Jörg Jödike, JL Software Solutions GbR.

**Ziel:** Entwicklung eines Tools (CLI) mit automatisiertem Workflows zur Erstellung spezialisierter, kleiner Sprachmodelle (TinyLlama, lokal) durch Zuführung von Dokumenten (PDF), die mit Hilfe eines großen, leistungsfähigen Modells trainiert werden (wiederholte Datenzuführung in einfachen Sätzen: Datendestillation aus der Quelle).

**Hauptelemente:**

0.  **Tool-Charakter:** Python-Skript mit Config-Datei (LLM-Pfade etc.) und Commands ("train" für Dokument einlesen, "query" für Abfragen)
1.  **Dokumenten-Einlesen:** Text aus verschiedenen Dokumentformaten extrahieren, insbesondere PDF.
2.  **Text-Zerlegung:** Dokumente in sinnvolle Abschnitte (Chunks) aufteilen.
3.  **Wissensdestillation:** Große Modelle nutzen, um aus den Textabschnitten vielfältige Sätze zu generieren, die das Wissen repräsentieren.
4.  **Datenaufbereitung:** Generierte Sätze für das Training des kleinen Modells formatieren.
5.  **Modelltraining:** Das kleine Modell (TinyLlama, lokal) mit den aufbereiteten Daten fine-tunen.

**Nutzen:**

*   Erstellung spezialisierter Sprachmodelle ohne manuelle Datenannotation.
*   Effiziente Nutzung von Ressourcen durch den Einsatz kleinerer Modelle.
*   Verbesserte Leistung kleiner Modelle durch die Kombination mit Wissen bzw. Sprachfähigkeit großer Modelle.

**Technologien:**

*   Python
*   PyPDF2 (für PDF-Verarbeitung)
*   Inference Modell zur Satzaufbereitung mit großem Modell, z.B. OpenRouter
*   Hugging Face Transformers (für kleine Modelle und Training)

**Hauptprämisse**

*   Nutzung von Free-Modellen, Konfiguration OpenRouter und Modellname
*   Lokalität: Das Trainingsmodell muss lokal gehalten werden
*   Das Trainingsmodell muss sich mit einem Sprachmodell in Kombination austauschen können, da das Trainingsmodell nur Spezialwissen enthält

**Erwartetes Ergebnis:** Ein funktionierendes Programm, das Dokumente verarbeitet und ein spezialisiertes TinyLlama-Modell erstellt, das in der Lage ist, Fragen zu den im Dokument enthaltenen Informationen zu beantworten.
