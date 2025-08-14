#!/usr/bin/env python3
"""
Test script for the training component to verify it works with modern TensorFlow 2.x APIs
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from CNNClassifier.config.configuration import ConfigurationManager
    from CNNClassifier.components.model_trainer import Training
    from CNNClassifier import logger
    
    print("✅ All imports successful!")
    
    # Test configuration loading
    config = ConfigurationManager()
    training_config = config.get_training_config()
    print("✅ Configuration loaded successfully!")
    
    # Test training component initialization
    training = Training(config=training_config)
    print("✅ Training component initialized successfully!")
    
    # Test base model loading
    training.get_base_model()
    print("✅ Base model loaded successfully!")
    
    # Test data generator creation
    training.train_valid_generator()
    print("✅ Data generators created successfully!")
    
    print("\n🎉 All tests passed! The training component is working correctly.")
    print("You can now run the full training pipeline using: python main.py")
    
except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
