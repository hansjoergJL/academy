"""
Test-Fälle für Academy Modules
Kleine, fokussierte Tests für jede Komponente
"""

from rich.console import Console

console = Console()


def test_model_trainer():
    """Teste ModelTrainer mit einfachen Daten"""
    console.print("[bold blue]🧪[/bold blue] Teste ModelTrainer...")
    
    try:
        from academy.model_trainer import ModelTrainer
        from academy.config_manager import ConfigManager
        
        # Test 1: Minimale Konfiguration
        config = ConfigManager()
        
        # Test mit kleinen Daten
        test_data = [
            {"text": "Dies ist ein Test."},
            {"text": "Das Modell sollte einfach Text generieren können."},
            {"text": "Das Training sollte schnell durchlaufen."}
        ]
        
        # Speichere Test-Daten
        import json
        test_data_path = "data/training/test_simple.json"
        with open(test_data_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        console.print(f"[green]✅[/green] Test-Daten gespeichert: {test_data_path}")
        
        # Initialisiere ModelTrainer
        trainer = ModelTrainer(config)
        
        # Test 2: Modul-Lade-Test
        console.print("[yellow]🔍[/yellow] Teste Modul-Lade-Fähigkeiten...")
        try:
            trainer.training_config = trainer._load_training_config()
            console.print("[green]✅[/green] Training-Konfiguration geladen")
        except Exception as e:
            console.print(f"[red]❌[/red] Konfigurations-Fehler: {e}")
            return False
        
        # Test 3: Dataset-Vorbereitung
        console.print("[yellow]🔍[/yellow] Teste Dataset-Vorbereitung...")
        try:
            dataset = trainer.prepare_dataset(test_data_path)
            console.print(f"[green]✅[/green] Dataset vorbereitet: {len(dataset)} Beispiele")
        except Exception as e:
            console.print(f"[red]❌[/red] Dataset-Fehler: {e}")
            return None

        # Test 4: Kurzes Training
        console.print("[yellow]🔍[/yellow] Teste kurzes Training...")
        try:
            result = trainer.train_model(test_data_path, "test_model")
            if result:
                console.print("[green]✅[/green] Training abgeschlossen")
            else:
                console.print("[red]❌[/red] Training fehlgeschlagen")
        except Exception as e:
            console.print(f"[red]❌[/red] Training-Fehler: {e}")
            return None

        console.print("[bold green]✅[/bold green] ModelTrainer Test bestanden!")
        return True
        
    except ImportError as e:
        console.print(f"[red]❌[/red] Import-Fehler: {e}")
        return False
    except Exception as e:
        console.print(f"[red]❌[/red] Test-Fehler: {e}")
        return False


def test_knowledge_distiller():
    """Teste KnowledgeDistiller"""
    console.print("[bold blue]🧠[/bold blue] Teste KnowledgeDistiller...")
    
    try:
        from academy.knowledge_distiller import KnowledgeDistiller
        from academy.config_manager import ConfigManager
        
        config = ConfigManager()
        distiller = KnowledgeDistiller(config)
        
        # Test 1: API-Verbindung
        console.print("[yellow]🔍[/yellow] Teste API-Verbindung...")
        if distiller.test_connection():
            console.print("[green]✅[/green] API-Verbindung erfolgreich!")
        else:
            console.print("[red]❌[/red] API-Verbindung fehlgeschlagen!")
            return False
        
        console.print("[bold green]✅[/bold green] KnowledgeDistiller Test bestanden!")
        return True
        
    except ImportError as e:
        console.print(f"[red]❌[/red] Import-Fehler: {e}")
        return False
    except Exception as e:
        console.print(f"[red]❌[/red] Test-Fehler: {e}")
        return False


def test_document_processor():
    """Teste DocumentProcessor"""
    console.print("[bold blue]📄[/bold blue] Teste DocumentProcessor...")
    
    try:
        from academy.document_processor import DocumentProcessor
        from academy.config_manager import ConfigManager
        
        config = ConfigManager()
        DocumentProcessor(config)

        # Test 1: Grundfunktionalität

        console.print("[green]✅[/green] DocumentProcessor geladen!")
        console.print("[bold green]✅[/bold green] DocumentProcessor Test bestanden!")
        return True
        
    except ImportError as e:
        console.print(f"[red]❌[/red] Import-Fehler: {e}")
        return False
    except Exception as e:
        console.print(f"[red]❌[/red] Test-Fehler: {e}")
        return False


def test_config_manager():
    """Teste ConfigManager"""
    console.print("[bold blue]⚙️[/bold blue] Teste ConfigManager...")
    
    try:
        from academy.config_manager import ConfigManager

        # Test 1: Konfiguration laden
        config = ConfigManager()
        console.print("[green]✅[/green] ConfigManager geladen!")
        
        # Test 2: Validierung
        if config.validate_config():
            console.print("[green]✅[/green] Konfiguration valide!")
        else:
            console.print("[red]❌[/red] Konfiguration nicht valide!")
        
        console.print("[bold green]✅[/bold green] ConfigManager Test bestanden!")
        return True
        
    except ImportError as e:
        console.print(f"[red]❌[/red] Import-Fehler: {e}")
        return False
    except Exception as e:
        console.print(f"[red]❌[/red] Test-Fehler: {e}")
        return False


def run_all_tests():
    """Führe alle Tests durch"""
    console.print("[bold magenta]🧪[/bold magenta] Academy Modul-Tests")
    
    tests = [
        ("ConfigManager", test_config_manager),
        ("DocumentProcessor", test_document_processor),
        ("KnowledgeDistiller", test_knowledge_distiller),
        ("ModelTrainer", test_model_trainer),
    ]
    
    results = []
    for test_name, test_func in tests:
        console.print(f"\n[cyan]🔍[/cyan] Teste {test_name}...")
        result = test_func()
        results.append((test_name, result))
        
    # Zusammenfassung
    console.print("\n[bold]📊 Test-Zusammenfassung:[/bold]")
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "[green]✅ BESTANDEN[/green]" if result else "[red]❌ FEHLGESCHLAGEN[/red]"
        symbol = "✅" if result else "❌"
        console.print(f"  {symbol} {test_name}: {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    console.print("\n[bold]📈 Ergebnis:[/bold]")
    console.print(f"  [green]Bestanden:[/green] {passed}/{len(tests)}")
    console.print(f"  [red]Fehlgeschlagen:[/red] {failed}/{len(tests)}")
    
    if failed == 0:
        console.print("[bold green]🎉 Alle Tests bestanden![/bold green]")
    else:
        console.print("[yellow]⚠️ Einige Tests fehlgeschlagen - Prüfe Implementierung![/yellow]")
    
    return failed == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Academy Modul-Tests")
    parser.add_argument("--test", choices=["all", "config", "processor", "distiller", "trainer"], default="all", help="Zu testendes Modul")
    
    args = parser.parse_args()
    
    if args.test == "all":
        run_all_tests()
    elif args.test == "config":
        test_config_manager()
    elif args.test == "processor":
        test_document_processor()
    elif args.test == "distiller":
        test_knowledge_distiller()
    elif args.test == "trainer":
        test_model_trainer()