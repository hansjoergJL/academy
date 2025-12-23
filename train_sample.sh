#!/bin/bash

# Academy Training Script für Fine-Tuning Guide Modell
# Dieses Skript führt das Training des Modells mit den vorbereiteten Daten durch.
# Kann für Fortsetzung verwendet werden, falls Checkpoints vorhanden sind.

echo "Starte Training des Fine-Tuning Guide Modells..."
echo "Daten: data/training/fine_tuning_guide_training.json"
echo "Modell: fine_tuning_guide"
echo ""

PYTHONPATH=. python -c "
from academy.config_manager import ConfigManager
from academy.model_trainer import ModelTrainer
config = ConfigManager()
trainer = ModelTrainer(config)
trainer.train_model('data/training/fine_tuning_guide_training.json', 'fine_tuning_guide')
"

echo ""
echo "Training abgeschlossen. Modell gespeichert in models/custom/fine_tuning_guide/"