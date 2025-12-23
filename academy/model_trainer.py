"""
Model Trainer für Academy
TinyLlama Fine-Tuning mit LoRA und HuggingFace
"""

import os
import sys
import time
import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, track
from rich.table import Table
from rich.panel import Panel

# Importiere Config Manager
sys.path.insert(0, str(Path(__file__).parent))
from config_manager import ConfigManager

console = Console()


@dataclass
class TrainingConfig:
    """Konfiguration für Model Training"""
    # LoRA Konfiguration
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    
    # Training Parameter
    batch_size: int = 1
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_length: int = 512
    gradient_accumulation_steps: int = 4
    
    # Hardware
    device: str = "auto"
    use_cpu: bool = False
    fp16: bool = True
    
    # Output
    output_dir: str = "./models/custom"
    logging_dir: str = "./logs"
    save_steps: int = 100
    save_total_limit: int = 3
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class TinyLlamaTrainer:
    """TinyLlama Fine-Tuning mit LoRA"""
    
    def __init__(self, config: TrainingConfig):
        """Initialisiere TinyLlamaTrainer"""
        self.config = config
        self.device = self._get_device()
        console.print(f"[green]✅[/green] Device: {self.device}")
        
        # Installiere fehlende Dependencies
        self._check_dependencies()
        
        # Setup Model und Tokenizer
        self._setup_model_and_tokenizer()
    
    def _get_device(self) -> str:
        """Auto device detection"""
        if self.config.use_cpu:
            return "cpu"
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    
    def _check_dependencies(self):
        """Prüfe und installiere Dependencies"""
        required_packages = [
            'transformers', 'peft', 'accelerate', 'datasets'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            console.print(f"[yellow]⚠️[/yellow] Fehlende Packages: {', '.join(missing_packages)}")
            console.print("[yellow]Installiere mit:[/yellow]")
            for package in missing_packages:
                install_cmd = f"pip install {package}"
                console.print(f"  [cyan]{install_cmd}[/cyan]")
            raise ImportError(f"Missing packages: {missing_packages}")
    
    def _setup_model_and_tokenizer(self):
        """Lade Modell und Tokenizer"""
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        console.print(f"[blue]🤖[/blue] Lade Modell: {model_name}")
        
        try:
            # Import im try-catch für bessere Fehlermeldungen
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import LoraConfig, get_peft_model, TaskType
            
            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
                # Model Konfiguration für Device (ohne device_map - nur dtype)
                model_kwargs = {
                    "dtype": torch.float16 if self.config.fp16 else torch.float32
                }
            
            # Quantisierung für GPU
            if self.device == "cuda":
                try:
                    model_kwargs["load_in_8bit"] = True
                    console.print("[green]✅[/green] 8-bit Quantisierung aktiviert")
                except:
                    console.print("[yellow]⚠️[/yellow] 8-bit nicht verfügbar, nutze Standard")
            
            # Model laden
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # LoRA Konfiguration
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            # LoRA Model anwenden
            self.model = get_peft_model(self.model, lora_config)
            
            # Trainable Parameters anzeigen
            self.model.print_trainable_parameters()
            
            console.print("[green]✅[/green] Modell und LoRA erfolgreich geladen")
            
        except ImportError as e:
            console.print(f"[red]❌[/red] Dependency Fehler: {e}")
            raise
        except Exception as e:
            console.print(f"[red]❌[/red] Modell-Lade Fehler: {e}")
            raise
    
    def prepare_dataset(self, training_data_path: str):
        """Dataset vorbereiten"""
        console.print(f"[blue]📊[/blue] Bereite Dataset vor: {training_data_path}")
        
        # Lade Trainingsdaten
        with open(training_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extrahiere Texte aus verschiedenen Formaten
        texts = []
        for item in data:
            if "text" in item:
                # Einfache Sätze Format
                texts.append(item["text"])
            elif "messages" in item:
                # Chat-Format zu Text konvertieren
                messages = item["messages"]
                text = ""
                for msg in messages:
                    if msg["role"] == "user":
                        text += f"User: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        text += f"Assistant: {msg['content']}\n"
                texts.append(text.strip())
        
        console.print(f"[green]✅[/green] {len(texts)} Trainingsbeispiele geladen")
        
        # Erstelle Dataset
        try:
            from datasets import Dataset
            
            dataset = Dataset.from_dict({"text": texts})
            
            # Tokenisierung
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.config.max_length,
                    return_overflowing_tokens=False,
                )
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            console.print(f"[green]✅[/green] Dataset tokenisiert ({len(tokenized_dataset)} Beispiele)")
            return tokenized_dataset
            
        except ImportError:
            console.print("[red]❌[/red] datasets package nicht verfügbar")
            raise
    
    def setup_training_arguments(self, output_dir: str):
        """Training Arguments konfigurieren"""
        try:
            from transformers import TrainingArguments
            
            return TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                logging_steps=10,
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
                eval_strategy="no",
                prediction_loss_only=True,
                remove_unused_columns=False,
                load_best_model_at_end=False,
                
                # Performance Optimierungen
                fp16=self.config.fp16 and self.device == "cuda",
                dataloader_num_workers=2,
                
                # Logging
                report_to="none",  # Kann auf "tensorboard" gesetzt werden
                disable_tqdm=False,
            )
        except ImportError:
            console.print("[red]❌[/red] transformers nicht verfügbar")
            raise
    
    def train(self, training_data_path: str, model_name: str):
        """Fine-Tuning durchführen"""
        console.print(f"[bold blue]🚀[/bold blue] Starte Fine-Tuning für {model_name}")
        
        # Output Verzeichnis
        output_dir = Path(self.config.output_dir) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Import im try-catch für bessere Fehlermeldungen  
            from transformers import Trainer, DataCollatorForLanguageModeling
            
            # Dataset vorbereiten
            dataset = self.prepare_dataset(training_data_path)
            
            # Training Arguments
            training_args = self.setup_training_arguments(str(output_dir))
            
            # Data Collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal Language Modeling
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )
            
            # Training starten
            console.print("[bold green]🎯[/bold green] Starte Training...")
            start_time = time.time()
            
            trainer.train()
            
            end_time = time.time()
            training_time = end_time - start_time
            console.print(f"[green]✅[/green] Training abgeschlossen in {training_time:.1f}s")
            
            # Modell speichern
            console.print(f"[blue]💾[/blue] Speichere Modell nach {output_dir}")
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Metadaten speichern
            metadata = {
                "model_name": model_name,
                "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "training_config": self.config.__dict__,
                "training_samples": len(dataset),
                "training_time_seconds": training_time,
                "device": self.device
            }
            
            with open(output_dir / "training_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            console.print("[bold green]✅[/bold green] Fine-Tuning erfolgreich abgeschlossen!")
            
            # Zeige Modellinformationen
            self._show_model_info(output_dir)
            
            return str(output_dir)
            
        except ImportError as e:
            console.print(f"[red]❌[/red] Training Dependencies: {e}")
            return None
        except Exception as e:
            console.print(f"[red]❌[/red] Training Fehler: {e}")
            return None
    
    def _show_model_info(self, model_path: Path):
        """Zeige Modellinformationen"""
        console.print("\n[bold]📋 Modell-Informationen:[/bold]")
        
        table = Table()
        table.add_column("Parameter", style="cyan")
        table.add_column("Wert", style="green")
        
        # Dateigröße
        model_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        table.add_row("Modellpfad", str(model_path))
        table.add_row("Größe", f"{model_size / 1024 / 1024:.1f} MB")
        table.add_row("Device", self.device)
        table.add_row("LoRA Rank", str(self.config.lora_rank))
        table.add_row("LoRA Alpha", str(self.config.lora_alpha))
        table.add_row("Epochs", str(self.config.num_epochs))
        table.add_row("Batch Size", str(self.config.batch_size))
        table.add_row("Learning Rate", f"{self.config.learning_rate}")
        
        console.print(table)


class ModelTrainer:
    """Hauptklasse für Model Training"""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialisiere ModelTrainer"""
        self.config_manager = config_manager
        self.training_config = self._load_training_config()
    
    def _load_training_config(self) -> TrainingConfig:
        """Lade Training-Konfiguration"""
        training_cfg = self.config_manager.get('training', {})
        lora_cfg = training_cfg.get('lora', {})
        
        return TrainingConfig(
            # LoRA
            lora_rank=lora_cfg.get('r', 8),
            lora_alpha=lora_cfg.get('alpha', 16),
            lora_dropout=lora_cfg.get('dropout', 0.05),
            target_modules=lora_cfg.get('target_modules', None),
            
            # Training
            batch_size=training_cfg.get('batch_size', 1),
            learning_rate=training_cfg.get('learning_rate', 2e-4),
            num_epochs=training_cfg.get('num_epochs', 3),
            max_length=training_cfg.get('max_length', 512),
            gradient_accumulation_steps=training_cfg.get('gradient_accumulation_steps', 4),
            
            # Hardware
            device=training_cfg.get('device', 'auto'),
            use_cpu=training_cfg.get('use_cpu', False),
            fp16=training_cfg.get('fp16', True),
            
            # Output
            output_dir=training_cfg.get('output_dir', './models/custom'),
            logging_dir=training_cfg.get('logging_dir', './logs'),
            save_steps=training_cfg.get('save_steps', 100),
            save_total_limit=training_cfg.get('save_total_limit', 3)
        )
    
    def train_model(self, training_data_path: str, model_name: str) -> Optional[str]:
        """Trainiere Modell mit gegebenen Daten"""
        console.print(Panel.fit(f"[bold blue]Model Training[/bold blue]\nModel: {model_name}\nDaten: {training_data_path}"))
        
        # Validiere Eingabedatei
        if not Path(training_data_path).exists():
            console.print(f"[red]❌[/red] Trainingsdaten nicht gefunden: {training_data_path}")
            return None
        
        # Erstelle Trainer
        trainer = TinyLlamaTrainer(self.training_config)
        
        # Führe Training durch
        result_path = trainer.train(training_data_path, model_name)
        
        return result_path
    
    def list_available_models(self):
        """Liste verfügbare Modelle auf"""
        models_dir = Path(self.training_config.output_dir)
        
        table = Table(title="Verfügbare Modelle")
        table.add_column("Modellname", style="cyan")
        table.add_column("Pfad", style="green")
        table.add_column("Größe", style="white")
        table.add_column("Erstellt", style="dim")
        
        if models_dir.exists():
            for model_path in models_dir.iterdir():
                if model_path.is_dir():
                    # Berechne Größe
                    model_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
                    size_str = f"{model_size / 1024 / 1024:.1f} MB"
                    
                    # Erstellzeit
                    mtime = model_path.stat().st_mtime
                    import datetime
                    created = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                    
                    table.add_row(model_path.name, str(model_path), size_str, created)
        else:
            console.print("[yellow]Keine Modelle gefunden[/yellow]")
            return
        
        console.print(table)


def main():
    """Test-Funktion für ModelTrainer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ModelTrainer Test')
    parser.add_argument('--training-data', '-i', required=True, help='Trainingsdaten JSON')
    parser.add_argument('--model-name', '-m', required=True, help='Modellname')
    parser.add_argument('--config', '-c', help='Konfigurationsdatei')
    
    args = parser.parse_args()
    
    # Initialisiere Config Manager
    config_manager = ConfigManager(args.config)
    
    # Initialisiere ModelTrainer
    trainer = ModelTrainer(config_manager)
    
    # Führe Training durch
    result_path = trainer.train_model(args.training_data, args.model_name)
    
    if result_path:
        console.print(f"[green]✅[/green] Modell gespeichert: {result_path}")
    else:
        console.print("[red]❌[/red] Training fehlgeschlagen")


if __name__ == "__main__":
    main()