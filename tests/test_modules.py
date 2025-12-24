#!/usr/bin/env python3
"""
Academy Module Tests
Testet alle Hauptmodule auf korrekte Funktionalität
"""

import sys
import os
from pathlib import Path

# Füge Academy zu Python Path hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Testet alle wichtigen Imports"""
    print("Testing imports...")

    try:
        import academy.config_manager
        import academy.document_processor
        import academy.knowledge_distiller
        import academy.model_trainer
        import academy.cli
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_manager():
    """Testet ConfigManager"""
    print("Testing ConfigManager...")

    try:
        from academy.config_manager import ConfigManager
        config = ConfigManager()
        # Test basic config access
        openrouter_config = config.get_openrouter_config()
        if 'api_key' in openrouter_config:
            print("✅ ConfigManager works")
            return True
        else:
            print("❌ ConfigManager missing keys")
            return False
    except Exception as e:
        print(f"❌ ConfigManager error: {e}")
        return False

def test_document_processor():
    """Testet Document Processor"""
    print("Testing Document Processor...")

    try:
        from academy.document_processor import DocumentProcessor
        from academy.config_manager import ConfigManager
        config_manager = ConfigManager()
        processor = DocumentProcessor(config_manager)
        # Test initialization
        print("✅ DocumentProcessor initialized")
        return True
    except Exception as e:
        print(f"❌ DocumentProcessor error: {e}")
        return False

def test_knowledge_distiller():
    """Testet Knowledge Distiller"""
    print("Testing Knowledge Distiller...")

    try:
        from academy.knowledge_distiller import KnowledgeDistiller
        from academy.config_manager import ConfigManager
        config = ConfigManager()
        distiller = KnowledgeDistiller(config)
        # Test initialization
        print("✅ KnowledgeDistiller initialized")
        return True
    except Exception as e:
        print(f"❌ KnowledgeDistiller error: {e}")
        return False

def test_model_trainer():
    """Testet Model Trainer"""
    print("Testing Model Trainer...")

    try:
        from academy.model_trainer import ModelTrainer
        from academy.config_manager import ConfigManager
        config = ConfigManager()
        trainer = ModelTrainer(config)
        # Test initialization
        print("✅ ModelTrainer initialized")
        return True
    except Exception as e:
        print(f"❌ ModelTrainer error: {e}")
        return False

def run_all_tests():
    """Führt alle Tests aus"""
    print("🧪 Running Academy Module Tests")
    print("=" * 40)

    tests = [
        ("Imports", test_imports),
        ("ConfigManager", test_config_manager),
        ("DocumentProcessor", test_document_processor),
        ("KnowledgeDistiller", test_knowledge_distiller),
        ("ModelTrainer", test_model_trainer)
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        print(f"\n🔍 {name}:")
        if test_func():
            passed += 1
        else:
            print(f"❌ {name} failed")

    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False

def main():
    """Hauptfunktion"""
    import argparse

    parser = argparse.ArgumentParser(description='Academy Module Tests')
    parser.add_argument('--test', choices=['all', 'imports', 'config', 'document', 'distiller', 'trainer'],
                       default='all', help='Which test to run')

    args = parser.parse_args()

    success = False
    if args.test == 'all':
        success = run_all_tests()
    elif args.test == 'imports':
        success = test_imports()
    elif args.test == 'config':
        success = test_config_manager()
    elif args.test == 'document':
        success = test_document_processor()
    elif args.test == 'distiller':
        success = test_knowledge_distiller()
    elif args.test == 'trainer':
        success = test_model_trainer()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()