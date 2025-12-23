#!/usr/bin/env python3
"""
Academy Environment Setup Script
Erstellt die notwendige Verzeichnisstruktur und konfiguriert die Umgebung
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil


def create_directory_structure():
    """Erstellt die notwendige Verzeichnisstruktur"""
    print("🏗️  Erstelle Verzeichnisstruktur...")
    
    directories = [
        "academy",
        "models",
        "models/custom",
        "models/hf_cache", 
        "models/checkpoints",
        "data",
        "data/raw",
        "data/processed", 
        "data/training",
        "logs",
        "tests"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    print("✅ Verzeichnisstruktur erstellt")


def create_init_files():
    """Erstellt __init__.py Dateien für Python Packages"""
    print("🐍 Erstelle Python Package Dateien...")
    
    init_files = [
        "academy/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('"""Academy Package"""\n__version__ = "0.1.0"\n')
            print(f"   ✅ {init_file}")


def check_python_version():
    """Prüft Python Version"""
    print("🐍 Prüfe Python Version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Fehler: Python 3.8+ wird benötigt")
        print(f"   Aktuelle Version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} gefunden")
    return True


def install_requirements():
    """Installiert Python Dependencies"""
    print("📦 Installiere Python Dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installiert")
    except subprocess.CalledProcessError as e:
        print(f"❌ Fehler bei Installation: {e}")
        return False
    
    return True


def setup_config():
    """Richtet Konfigurationsdatei ein"""
    print("⚙️  Richte Konfiguration ein...")
    
    config_template = "config.yaml.template"
    config_file = "config.yaml"
    
    if not os.path.exists(config_file) and os.path.exists(config_template):
        shutil.copy(config_template, config_file)
        print(f"✅ {config_file} erstellt")
        print("⚠️  Bitte bearbeiten Sie config.yaml mit Ihren Einstellungen")
    elif os.path.exists(config_file):
        print(f"✅ {config_file} existiert bereits")
    else:
        print(f"❌ {config_template} nicht gefunden")


def create_env_file():
    """Erstellt .env Datei für Environment Variablen"""
    print("🌍 Erstelle Environment Datei...")
    
    env_file = ".env"
    if not os.path.exists(env_file):
        with open(env_file, 'w') as f:
            f.write("""# Academy Environment Variables
# OpenRouter API Configuration
OPENROUTER_API_KEY=your-openrouter-api-key-here

# HuggingFace Token (optional, für private Models)
HUGGINGFACE_TOKEN=your-huggingface-token-here

# Additional Environment Variables
ACADEMY_DATA_DIR=./data
ACADEMY_MODELS_DIR=./models
ACADEMY_LOGS_DIR=./logs
""")
        print(f"✅ {env_file} erstellt")
        print("⚠️  Bitte bearbeiten Sie .env mit Ihren API Keys")
    else:
        print(f"✅ {env_file} existiert bereits")


def create_gitignore():
    """Erstellt .gitignore Datei"""
    print("📝 Erstelle .gitignore...")
    
    gitignore_file = ".gitignore"
    gitignore_content = """# Python
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

# IDE
.vscode/
.idea/
*.swp
*.swo

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

# Jupyter Notebook
.ipynb_checkpoints
"""
    
    with open(gitignore_file, 'w') as f:
        f.write(gitignore_content)
    print(f"✅ {gitignore_file} erstellt")


def verify_setup():
    """Überprüft das Setup"""
    print("🔍 Überprüfe Setup...")
    
    checks = []
    
    # Prüfe wichtige Dateien
    important_files = [
        "requirements.txt",
        "config.yaml.template", 
        "README.md",
        "setup_environment.py"
    ]
    
    for file in important_files:
        if os.path.exists(file):
            checks.append(f"✅ {file}")
        else:
            checks.append(f"❌ {file}")
    
    # Prüfe Verzeichnisse
    important_dirs = [
        "academy/",
        "data/",
        "models/",
        "logs/"
    ]
    
    for directory in important_dirs:
        if os.path.exists(directory):
            checks.append(f"✅ {directory}")
        else:
            checks.append(f"❌ {directory}")
    
    for check in checks:
        print(f"   {check}")


def show_next_steps():
    """Zeigt die nächsten Schritte"""
    print("\n" + "="*60)
    print("🎉 Academy Environment Setup abgeschlossen!")
    print("="*60)
    print("\n📋 Nächste Schritte:")
    print("1. API Keys konfigurieren:")
    print("   - Bearbeiten Sie .env mit Ihrem OpenRouter API Key")
    print("   - Optional: HuggingFace Token hinzufügen")
    print("\n2. Konfiguration anpassen:")
    print("   - Bearbeiten Sie config.yaml nach Bedarf")
    print("   - Prüfen Sie Pfade und Modelleinstellungen")
    print("\n3. Entwicklung starten:")
    print("   - cd academy")
    print("   - Beginnen Sie mit der Implementierung von cli.py")
    print("\n4. CLI installieren (nach Implementierung):")
    print("   - pip install -e .")
    print("\n📚 Weitere Informationen:")
    print("   - README.md für detaillierte Anleitung")
    print("   - config.yaml.template für Konfigurationsoptionen")


def main():
    """Hauptfunktion"""
    print("🚀 Academy Environment Setup")
    print("="*40)
    
    # Python Version prüfen
    if not check_python_version():
        sys.exit(1)
    
    # Verzeichnisstruktur erstellen
    create_directory_structure()
    
    # Python Package Dateien erstellen
    create_init_files()
    
    # Konfiguration einrichten
    setup_config()
    
    # Environment Datei erstellen
    create_env_file()
    
    # .gitignore erstellen
    create_gitignore()
    
    # Dependencies installieren (optional)
    install_prompt = input("\n📦 Möchten Sie die Python Dependencies installieren? (y/N): ")
    if install_prompt.lower() in ['y', 'yes']:
        if not install_requirements():
            print("⚠️  Installation fehlgeschlagen. Installieren Sie manuell:")
            print("   pip install -r requirements.txt")
    
    # Setup überprüfen
    verify_setup()
    
    # Nächste Schritte zeigen
    show_next_steps()


if __name__ == "__main__":
    main()