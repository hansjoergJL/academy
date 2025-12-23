"""
CLI Interface für Academy
Hauptkommandozeilen-Interface mit allen Commands
"""

import click
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

# Füge Academy zu Python Path hinzu
sys.path.insert(0, str(Path(__file__).parent))

from config_manager import get_config, ConfigManager

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="Academy")
@click.option('--config', '-c', help='Pfad zur Konfigurationsdatei')
@click.pass_context
def cli(ctx, config):
    """Academy - Wissensdestillation für Spezialmodelle"""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config


@cli.command()
@click.pass_context
def init(ctx):
    """Initialisiere Academy Projekt"""
    console.print(Panel.fit("[bold blue]Academy Initialisierung[/bold blue]"))
    
    console.print("\n🔧 [yellow]Konfiguration einrichten...[/yellow]")
    config_manager = ConfigManager(ctx.obj.get('config_path'))
    
    # Interaktive Konfiguration
    console.print("\n🌐 [bold]OpenRouter API Konfiguration[/bold]")
    
    # API Key
    api_key_env = config_manager.get('openrouter.api_key_env', 'OPENROUTER_API_KEY')
    console.print(f"Environment Variable für API Key: {api_key_env}")
    
    if Confirm.ask("Möchten Sie den API Key direkt eingeben?", default=False):
        api_key = Prompt.ask("OpenRouter API Key", password=True)
        config_manager.set('openrouter.api_key', api_key)
    else:
        console.print(f"[green]✓[/green] Setzen Sie die Environment Variable {api_key_env}")
    
    # Modellname
    current_model = config_manager.get('openrouter.model_name')
    model_name = Prompt.ask(f"Modellname", default=current_model)
    config_manager.set('openrouter.model_name', model_name)
    
    # HuggingFace Token
    console.print("\n🤗 [bold]HuggingFace Konfiguration (optional)[/bold]")
    if Confirm.ask("HuggingFace Token konfigurieren?", default=False):
        hf_token = Prompt.ask("HuggingFace Token", password=True)
        config_manager.set('huggingface.token', hf_token)
    
    # Speichern
    config_manager.save_config()
    
    # Validieren
    if config_manager.validate_config():
        console.print("\n[green]✅ Academy erfolgreich initialisiert![/green]")
        console.print("Mit 'academy train --input <pdf_datei>' können Sie beginnen.")
    else:
        console.print("\n[red]❌ Initialisierung fehlgeschlagen![/red]")


@cli.command()
@click.option('--setup', is_flag=True, help='Interaktive Konfiguration')
@click.option('--validate', is_flag=True, help='Konfiguration validieren')
@click.option('--show', is_flag=True, help='Konfiguration anzeigen')
@click.pass_context
def config(ctx, setup, validate, show):
    """Konfiguration verwalten"""
    config_manager = ConfigManager(ctx.obj.get('config_path'))
    
    if setup:
        config_manager.interactive_setup()
    elif validate:
        if config_manager.validate_config():
            console.print("[green]✅ Konfiguration valide[/green]")
        else:
            console.print("[red]❌ Konfiguration fehlerhaft[/red]")
    elif show:
        # Zeige wichtige Konfiguration in Tabelle
        table = Table(title="Academy Konfiguration")
        table.add_column("Bereich", style="cyan")
        table.add_column("Parameter", style="green")
        table.add_column("Wert", style="white")
        
        # OpenRouter
        openrouter = config_manager.get_openrouter_config()
        table.add_row("OpenRouter", "Modell", openrouter['model_name'])
        table.add_row("OpenRouter", "Base URL", openrouter['base_url'])
        table.add_row("OpenRouter", "API Key", "***" if openrouter['api_key'] else "NICHT GEFUNDEN")
        
        # HuggingFace
        hf = config_manager.get_huggingface_config()
        table.add_row("HuggingFace", "Base Model", hf['base_model'])
        table.add_row("HuggingFace", "Token", "***" if hf.get('token') else "Nicht gesetzt")
        
        # Training
        training = config_manager.get_training_config()
        table.add_row("Training", "Batch Size", str(training.get('batch_size', 1)))
        table.add_row("Training", "Epochs", str(training.get('num_epochs', 3)))
        table.add_row("Training", "Device", training.get('device', 'auto'))
        
        # Pfade
        paths = config_manager.get_paths_config()
        table.add_row("Pfade", "Daten", paths.get('data_dir', './data'))
        table.add_row("Pfade", "Modelle", paths.get('models_dir', './models'))
        
        console.print(table)
    else:
        console.print("Verwenden Sie --setup, --validate oder --show")


@cli.command()
@click.option('--input', '-i', required=True, help='PDF Dokument oder Verzeichnis')
@click.option('--model-name', '-m', help='Name für das trainierte Modell')
@click.option('--output', '-o', help='Ausgabeverzeichnis (überschreibt config)')
@click.option('--overwrite', is_flag=True, help='Vorhandenes Modell überschreiben')
@click.option('--skip-distillation', is_flag=True, help='Überspringe Wissensdestillation')
@click.option('--cl', is_flag=True, help='Nur existierende Trainingsdaten nutzen, ohne neue Destillation')
@click.pass_context
def train(ctx, input, model_name, output, overwrite, skip_distillation, cl):
    """Trainiere ein spezialisiertes Modell aus PDF-Dokumenten"""
    config_manager = ConfigManager(ctx.obj.get('config_path'))
    
    console.print(Panel.fit(f"[bold blue]Training[/bold blue]\nInput: {input}"))
    
    # Validiere Konfiguration
    if not config_manager.validate_config():
        console.print("[red]❌ Konfiguration nicht valide![/red]")
        return
    
    # Prüfe Input
    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]❌ Input nicht gefunden: {input}[/red]")
        return
    
    # Modellname
    if not model_name:
        model_name = Prompt.ask("Modellname", default="custom-specialist")
    
    console.print(f"[green]✓[/green] Starte Training für Modell: {model_name}")
    
    paths = config_manager.get_paths_config()
    
    try:
        from document_processor import DocumentProcessor
        from knowledge_distiller import KnowledgeDistiller
        
        # 1. Dokumentenverarbeitung
        console.print("\n[blue]📄[/blue] Phase 1: Dokumentenverarbeitung")
        processor = DocumentProcessor(config_manager)
        
        if input_path.is_file():
            chunks = processor.process_file(input_path)
        else:
            chunks = processor.process_directory(input_path)
        
        # Speichere Chunks
        chunks_path = Path(paths['processed_chunks']) / f"{model_name}_chunks.json"
        processor.save_chunks(chunks, chunks_path)
        
        if cl:
            # --cl Option: Nutze existierende Trainingsdaten
            console.print("\n[blue]🤖[/blue] Phase 3: Training mit existierenden Daten (--cl)")
            from model_trainer import ModelTrainer
            
            trainer = ModelTrainer(config_manager)
            
            # Finde existierende Trainingsdaten
            training_data_files = list(Path(paths['training_data']).glob("*_training.json"))
            
            if not training_data_files:
                console.print("[red]❌ Keine Trainingsdaten gefunden![/red]")
                console.print(f"[yellow]💡[/yellow] Suche in: {paths['training_data']}")
                return
            
            # Verwende neueste Trainingsdaten
            latest_training = max(training_data_files, key=lambda f: f.stat().st_mtime)
            console.print(f"[green]✅[/green] Verwende Trainingsdaten: {latest_training}")
            
            # Führe Training durch
            result_path = trainer.train_model(str(latest_training), model_name)
            
            if result_path:
                console.print(f"[green]✅[/green] Training abgeschlossen: {result_path}")
            else:
                console.print("[red]❌[/red] Training fehlgeschlagen")
        else:
            # 2. Wissensdestillation
            console.print("\n[blue]🧠[/blue] Phase 2: Wissensdestillation")
            distiller = KnowledgeDistiller(config_manager)
            
            # Teste API Verbindung
            if not distiller.test_connection():
                console.print("[red]❌ API Verbindung fehlgeschlagen[/red]")
                return
            
            # Führe Destillation durch
            examples_path = Path(paths['training_data']) / f"{model_name}_examples.json"
            examples = distiller.distill_chunks(chunks, examples_path)
            
            # Formattiere für Training
            training_path = Path(paths['training_data']) / f"{model_name}_training.json"
            distiller.format_for_training(examples, training_path)
            
            # 3. Training (placeholder) - Normale Pipeline
            console.print(f"\n[blue]🤖[/blue] Phase 3: Modelltraining")
            console.print(f"[green]✅[/green] Vorbereitung abgeschlossen")
            console.print(f"[yellow]📊[/yellow] {len(chunks)} Chunks → {len(examples)} Trainingsbeispiele")
            console.print(f"[yellow]💾[/yellow] Trainingsdaten: {training_path}")
            console.print("[yellow]⚠️  Modelltraining wird in nächster Version implementiert[/yellow]")
        
    except Exception as e:
        console.print(f"[red]❌[/red] Training fehlgeschlagen: {e}")
        return


@cli.command()
@click.option('--model-name', '-m', required=True, help='Name des abzufragenden Modells')
@click.option('--interactive', '-i', is_flag=True, help='Interaktiver Abfragemodus')
@click.pass_context
def query(ctx, model_name, interactive):
    """Frage ein trainiertes Modell ab"""
    config_manager = ConfigManager(ctx.obj.get('config_path'))
    
    console.print(Panel.fit(f"[bold blue]Query[/bold blue]\nModell: {model_name}"))
    
    # TODO: Implementiere Abfragelogik
    console.print("[yellow]⚠️  Query-Logik noch nicht implementiert[/yellow]")


@cli.command()
@click.pass_context
def list_models(ctx):
    """Liste verfügbare Modelle auf"""
    config_manager = ConfigManager(ctx.obj.get('config_path'))
    paths = config_manager.get_paths_config()
    
    models_dir = Path(paths['models_dir'])
    
    table = Table(title="Verfügbare Modelle")
    table.add_column("Modellname", style="cyan")
    table.add_column("Pfad", style="green")
    table.add_column("Größe", style="white")
    table.add_column("Erstellt", style="dim")
    
    if models_dir.exists():
        # Suche nach Modell-Verzeichnissen
        for model_path in models_dir.iterdir():
            if model_path.is_dir() and model_path.name != 'hf_cache':
                # TODO: Ermittle Modellinformationen
                table.add_row(model_path.name, str(model_path), "N/A", "N/A")
    else:
        console.print("[yellow]Keine Modelle gefunden[/yellow]")
        return
    
    console.print(table)


@cli.command()
@click.argument('path', type=click.Path(exists=True))
def check_pdf(path):
    """Prüfe PDF auf Lesbarkeit"""
    try:
        from document_processor import DocumentProcessor
        
        config_manager = get_config()
        processor = DocumentProcessor(config_manager)
        
        pdf_path = Path(path)
        if pdf_path.suffix.lower() != '.pdf':
            console.print("[red]❌ Keine PDF-Datei[/red]")
            return
        
        console.print(f"[green]✓[/green] Prüfe PDF: {pdf_path}")
        
        # Verarbeite PDF
        chunks = processor.process_file(pdf_path)
        console.print(f"[green]✅[/green] PDF verarbeitbar: {len(chunks)} Chunks extrahiert")
        
        # Zeige erste paar Zeilen
        if chunks:
            console.print(f"\n[bold]Vorschau (erste 200 Zeichen):[/bold]")
            console.print(chunks[0].text[:200] + "..." if len(chunks[0].text) > 200 else chunks[0].text)
        
    except Exception as e:
        console.print(f"[red]❌[/red] Fehler bei PDF-Prüfung: {e}")


@cli.command()
def test_api():
    """Teste OpenRouter API Verbindung"""
    try:
        from knowledge_distiller import KnowledgeDistiller
        
        config_manager = get_config()
        distiller = KnowledgeDistiller(config_manager)
        
        console.print("[green]✓[/green] Teste OpenRouter API Verbindung...")
        
        # Teste API Verbindung
        if distiller.test_connection():
            console.print("[green]✅[/green] API Verbindung erfolgreich!")
            console.print(f"[dim]Modell: {distiller.model_name}[/dim]")
        else:
            console.print("[red]❌[/red] API Verbindung fehlgeschlagen")
        
    except Exception as e:
        console.print(f"[red]❌[/red] Fehler bei API-Test: {e}")


def main():
    """Hauptfunktion"""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Auf Wiedersehen![/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]❌[/red] Fehler: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()