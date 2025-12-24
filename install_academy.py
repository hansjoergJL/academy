#!/usr/bin/env python3
"""
Academy Full Installation Script
Automatisches Setup des Academy-Projekts auf einem neuen Server
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, description):
    """Führt ein Kommando aus und zeigt Ergebnis"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} erfolgreich")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Fehler bei {description}: {e}")
        print(f"   Output: {e.output}")
        return False


def check_python_version():
    """Prüft Python Version"""
    print("🐍 Prüfe Python Version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ benötigt, gefunden: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} gefunden")
    return True


def check_required_files():
    """Prüft, ob alle erforderlichen Dateien vorhanden sind"""
    print("📁 Prüfe erforderliche Dateien...")
    required_files = [
        "requirements.txt",
        "setup.py",
        "config.yaml.template",
        "README.md",
        "TECH.md",
        "AGENTS.md",
        "install_academy.py",
        "setup_environment.py",
        "academy/__init__.py",
        "academy/cli.py",
        "academy/config_manager.py",
        "academy/model_trainer.py",
        "academy/document_processor.py",
        "academy/knowledge_distiller.py",
        "tests/__init__.py",
        "tests/test_modules.py"
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print("❌ Fehlende Dateien:")
        for f in missing:
            print(f"   - {f}")
        return False

    print("✅ Alle erforderlichen Dateien vorhanden")
    return True


def install_dependencies():
    """Installiert Python-Abhängigkeiten"""
    if not run_command(f"{sys.executable} -m pip install --upgrade pip",
                       "Upgrade pip"):
        return False

    if not run_command(f"{sys.executable} -m pip install -r requirements.txt",
                       "Installiere Python-Abhängigkeiten"):
        return False

    return True


def setup_environment():
    """Führt das Environment Setup aus"""
    print("🏗️  Richte Umgebung ein...")
    try:
        # Führe setup_environment.py aus, aber ohne interactive prompts
        # Wir simulieren die Eingabe
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())

        # Erstelle Verzeichnisse manuell
        directories = [
            "academy", "models", "models/custom", "models/hf_cache",
            "models/checkpoints", "data", "data/raw", "data/processed",
            "data/training", "logs", "tests"
        ]
        for d in directories:
            Path(d).mkdir(parents=True, exist_ok=True)

        # Erstelle __init__.py
        for init_file in ["academy/__init__.py", "tests/__init__.py"]:
            if not Path(init_file).exists():
                Path(init_file).write_text('"""Academy Package"""\n__version__ = "0.1.0"\n')

        # Kopiere config
        if Path("config.yaml.template").exists() and not Path("config.yaml").exists():
            shutil.copy("config.yaml.template", "config.yaml")

        # Erstelle .env
        if not Path(".env").exists():
            Path(".env").write_text("""# Academy Environment Variables
# OpenRouter API Configuration
OPENROUTER_API_KEY=your-openrouter-api-key-here

# HuggingFace Token (optional)
HUGGINGFACE_TOKEN=your-huggingface-token-here

# Paths
ACADEMY_DATA_DIR=./data
ACADEMY_MODELS_DIR=./models
ACADEMY_LOGS_DIR=./logs
""")

        # Erstelle .gitignore
        if not Path(".gitignore").exists():
            Path(".gitignore").write_text("""# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/

# Academy Specific
config.yaml
.env
logs/
models/custom/
models/hf_cache/
models/checkpoints/
data/raw/
data/processed/
data/training/
*.log

# OS
.DS_Store
Thumbs.db
""")

        print("✅ Umgebung eingerichtet")
        return True

    except Exception as e:
        print(f"❌ Fehler beim Setup: {e}")
        return False


def install_cli():
    """Installiert die Academy CLI"""
    return run_command(f"{sys.executable} -m pip install -e .",
                       "Installiere Academy CLI")


def run_tests():
    """Führt Basis-Tests aus"""
    print("🧪 Führe Basis-Tests aus...")
    try:
        # Import-Test
        sys.path.insert(0, str(Path.cwd()))
        import academy.config_manager
        import academy.cli
        print("✅ Imports erfolgreich")

        # Config-Test
        config = academy.config_manager.ConfigManager()
        if config.validate_config():
            print("✅ Konfiguration valide")
            return True
        else:
            print("⚠️  Konfiguration unvollständig (API Keys fehlen)")
            return True  # Nicht kritisch für Installation

    except Exception as e:
        print(f"❌ Test-Fehler: {e}")
        return False


def main():
    """Hauptinstallation"""
    print("🚀 Academy Full Installation")
    print("=" * 50)

    success = True

    # 1. Python-Version
    if not check_python_version():
        success = False

    # 2. Erforderliche Dateien
    if not check_required_files():
        success = False

    # 3. Dependencies
    if success and not install_dependencies():
        success = False

    # 4. Environment Setup
    if success and not setup_environment():
        success = False

    # 5. CLI Installation
    if success and not install_cli():
        success = False

    # 6. Tests
    if success and not run_tests():
        print("⚠️  Tests fehlgeschlagen, aber Installation fortgesetzt")

    # Abschluss
    print("\n" + "=" * 50)
    if success:
        print("🎉 Academy erfolgreich installiert!")
        print("\n📋 Nächste Schritte:")
        print("1. API Keys konfigurieren:")
        print("   - Bearbeiten Sie .env und config.yaml")
        print("2. Dokumente in data/raw/ legen")
        print("3. Training starten: academy train --input data/raw/ --model-name test")
        print("\n📚 Dokumentation:")
        print("   - README.md für Benutzeranleitung")
        print("   - TECH.md für technische Details")
    else:
        print("❌ Installation fehlgeschlagen!")
        print("Prüfen Sie die Fehler oben und versuchen Sie es erneut.")
        sys.exit(1)


if __name__ == "__main__":
    main()