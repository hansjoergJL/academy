"""
Configuration Manager für Academy
Verwaltet alle Konfigurationseinstellungen aus config.yaml und Environment Variablen
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Import dotenv - install if not available
try:
    from dotenv import load_dotenv
except ImportError:
    print("❌ python-dotenv nicht installiert. Bitte installieren mit:")
    print("   pip install python-dotenv")
    sys.exit(1)


class ConfigManager:
    """Zentrale Konfigurationsverwaltung"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialisiere Config Manager"""
        # Lade Environment Variablen
        load_dotenv()
        
        # Setze Konfigurationspfad
        if config_path is None:
            # Projektverzeichnis finden
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path) if config_path else None
        self._config: Dict[str, Any] = {}
        
        # Lade Konfiguration
        self.load_config()
    
    def load_config(self) -> None:
        """Lade Konfiguration aus YAML Datei"""
        try:
            if self.config_path and self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f) or {}
                print(f"✅ Konfiguration geladen: {self.config_path}")
            else:
                print(f"⚠️  Konfigurationsdatei nicht gefunden: {self.config_path}")
                print("   Erstelle Standardkonfiguration...")
                self._config = self._get_default_config()
                self.save_config()
        except Exception as e:
            print(f"❌ Fehler beim Laden der Konfiguration: {e}")
            self._config = self._get_default_config()
    
    def save_config(self) -> None:
        """Speichere Konfiguration in YAML Datei"""
        try:
            if self.config_path:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self._config, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
                print(f"✅ Konfiguration gespeichert: {self.config_path}")
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Konfiguration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Hole Konfigurationswert mit dot-notation (z.B. 'openrouter.api_key')"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                # Versuche Environment Variable
                env_key = key.upper().replace('.', '_')
                env_value = os.getenv(env_key)
                if env_value:
                    return env_value
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Setze Konfigurationswert mit dot-notation"""
        keys = key.split('.')
        config = self._config
        
        # Navigiere zum Ziel
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Setze Wert
        config[keys[-1]] = value
    
    def get_openrouter_config(self) -> Dict[str, Any]:
        """Hole OpenRouter Konfiguration"""
        return {
            'api_key': self._get_api_key(),
            'model_name': self.get('openrouter.model_name', 'meta-llama/llama-3.2-3b-instruct:free'),
            'base_url': self.get('openrouter.base_url', 'https://openrouter.ai/api/v1'),
            'timeout': self.get('openrouter.timeout', 30),
            'max_retries': self.get('openrouter.max_retries', 3),
            'rate_limit_delay': self.get('openrouter.rate_limit_delay', 1.0)
        }
    
    def _get_api_key(self) -> str:
        """Hole OpenRouter API Key"""
        # Zuerst versuchen Environment Variable
        api_key_env = self.get('openrouter.api_key_env', 'OPENROUTER_API_KEY')
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            # Direkten API Key versuchen
            api_key = self.get('openrouter.api_key')
        
        if not api_key:
            raise ValueError(
                f"OpenRouter API Key nicht gefunden! "
                f"Setzen Sie die Environment Variable {api_key_env} oder "
                f"konfigurieren Sie openrouter.api_key in config.yaml"
            )
        
        return api_key
    
    def get_huggingface_config(self) -> Dict[str, Any]:
        """Hole HuggingFace Konfiguration"""
        token_env = self.get('huggingface.token_env', 'HUGGINGFACE_TOKEN')
        token = os.getenv(token_env)
        
        return {
            'token': token,
            'base_model': self.get('huggingface.base_model', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'),
            'cache_dir': self.get('huggingface.cache_dir', './models/hf_cache'),
            'push_to_hub': self.get('huggingface.push_to_hub', False),
            'repository_name': self.get('huggingface.repository_name', ''),
            'private': self.get('huggingface.private', False)
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Hole Training Konfiguration"""
        return {
            'lora': self.get('training.lora', {}),
            'batch_size': self.get('training.batch_size', 1),
            'learning_rate': self.get('training.learning_rate', 2e-4),
            'num_epochs': self.get('training.num_epochs', 3),
            'max_length': self.get('training.max_length', 512),
            'device': self.get('training.device', 'auto'),
            'use_cpu': self.get('training.use_cpu', False),
            'output_dir': self.get('training.output_dir', './models/custom'),
            'logging_dir': self.get('training.logging_dir', './logs')
        }
    
    def get_documents_config(self) -> Dict[str, Any]:
        """Hole Dokumentenverarbeitungs-Konfiguration"""
        return {
            'supported_formats': self.get('documents.supported_formats', ['.pdf', '.txt', '.md']),
            'pdf': self.get('documents.pdf', {}),
            'chunking': self.get('documents.chunking', {})
        }
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Hole Pfade Konfiguration"""
        return {
            'data_dir': self.get('paths.data_dir', './data'),
            'raw_documents': self.get('paths.raw_documents', './data/raw'),
            'processed_chunks': self.get('paths.processed_chunks', './data/processed'),
            'training_data': self.get('paths.training_data', './data/training'),
            'models_dir': self.get('paths.models_dir', './models'),
            'checkpoints_dir': self.get('paths.checkpoints_dir', './models/checkpoints'),
            'logs_dir': self.get('paths.logs_dir', './logs')
        }
    
    def validate_config(self) -> bool:
        """Validiere Konfiguration"""
        errors = []
        
        # Prüfe OpenRouter API Key
        try:
            self._get_api_key()
        except ValueError as e:
            errors.append(str(e))
        
        # Prüfe Verzeichnisse
        paths = self.get_paths_config()
        for path_key, path_value in paths.items():
            path = Path(path_value)
            if not path.exists() and path_key in ['data_dir', 'models_dir', 'logs_dir']:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Konnte Verzeichnis nicht erstellen: {path} - {e}")
        
        if errors:
            print("❌ Konfigurationsfehler:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        print("✅ Konfiguration validiert")
        return True
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Standardkonfiguration"""
        return {
            'openrouter': {
                'api_key_env': 'OPENROUTER_API_KEY',
                'model_name': 'meta-llama/llama-3.2-3b-instruct:free',
                'base_url': 'https://openrouter.ai/api/v1',
                'timeout': 30,
                'max_retries': 3,
                'rate_limit_delay': 1.0
            },
            'huggingface': {
                'token_env': 'HUGGINGFACE_TOKEN',
                'base_model': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                'cache_dir': './models/hf_cache',
                'push_to_hub': False,
                'repository_name': '',
                'private': False
            },
            'training': {
                'lora': {
                    'r': 8,
                    'alpha': 16,
                    'dropout': 0.05,
                    'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
                },
                'batch_size': 1,
                'learning_rate': 2e-4,
                'num_epochs': 3,
                'max_length': 512,
                'device': 'auto',
                'use_cpu': False,
                'output_dir': './models/custom',
                'logging_dir': './logs'
            },
            'paths': {
                'data_dir': './data',
                'raw_documents': './data/raw',
                'processed_chunks': './data/processed',
                'training_data': './data/training',
                'models_dir': './models',
                'checkpoints_dir': './models/checkpoints',
                'logs_dir': './logs'
            }
        }
    
    def interactive_setup(self) -> None:
        """Interaktive Konfigurationseinrichtung"""
        print("🔧 Academy Konfiguration einrichten")
        print("=" * 50)
        
        # OpenRouter API Key
        print("\n🌐 OpenRouter API Konfiguration")
        api_key = input("OpenRouter API Key (oder ENTER für Environment Variable): ").strip()
        if api_key:
            self.set('openrouter.api_key', api_key)
        
        model_name = input(f"Modellname (default: {self.get('openrouter.model_name')}): ").strip()
        if model_name:
            self.set('openrouter.model_name', model_name)
        
        # HuggingFace Token
        print("\n🤗 HuggingFace Konfiguration (optional)")
        hf_token = input("HuggingFace Token (optional für private Models): ").strip()
        if hf_token:
            self.set('huggingface.token', hf_token)
        
        # Pfade anpassen
        print("\n📁 Pfade (default wird empfohlen)")
        base_dir = input(f"Basisverzeichnis für Daten (default: {self.get('paths.data_dir')}): ").strip()
        if base_dir:
            self.set('paths.data_dir', base_dir)
        
        # Speichern
        self.save_config()
        print("\n✅ Konfiguration gespeichert!")
        
        # Validieren
        self.validate_config()


# Globale Config Instance
_config_manager = None

def get_config() -> ConfigManager:
    """Globale Config Instanz"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def main():
    """Test/Setup der Konfiguration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Academy Configuration Manager')
    parser.add_argument('--setup', action='store_true', help='Interaktive Konfiguration')
    parser.add_argument('--validate', action='store_true', help='Konfiguration validieren')
    parser.add_argument('--show', action='store_true', help='Konfiguration anzeigen')
    
    args = parser.parse_args()
    
    if args.setup:
        config = ConfigManager()
        config.interactive_setup()
    
    if args.validate:
        config = ConfigManager()
        config.validate_config()
    
    if args.show:
        config = ConfigManager()
        print("Aktuelle Konfiguration:")
        print(yaml.dump(config._config, default_flow_style=False, allow_unicode=True, indent=2))


if __name__ == "__main__":
    main()