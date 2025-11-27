#!/usr/bin/env python3
import os
import sys
import shutil

# Add the parent directory to the path to import the file1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from file1.config import File1Config
from file1.file_manager import FileManager

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_repo_dir = os.path.join(current_dir, "test_repo")
    outputs_dir = os.path.join(current_dir, "outputs")
    config_path = os.path.join(current_dir, "..", "config.toml")
    
    # Create outputs directory if it doesn't exist
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Copy test_repo to outputs
    target_dir = os.path.join(outputs_dir, "test_repo")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(test_repo_dir, target_dir)
    print(f"Copied test_repo to {target_dir}")
    
    # Load configuration
    config = File1Config.from_toml(config_path)
    print(f"Loaded configuration from {config_path}")
    
    # Initialize file manager
    file_manager = FileManager(
        config=config,
        analyze_dir=target_dir,
        summary_cache_path=os.path.join(outputs_dir, "file_summary_cache.json"),
        backup_path=os.path.join(outputs_dir, "backup")
    )
    
    # Clean the repository (remove duplicates and simulated data)
    deleted_files = file_manager.clean_repository()
    print(f"Repository cleaning completed. Deleted {len(deleted_files)} files")
    
    # Print details about deleted files
    for file_info in deleted_files:
        print(f"Deleted: {os.path.relpath(file_info['path'], target_dir)} - Reason: {file_info['reason']}")
    
    # Build file relationship graph
    file_manager.build_graph()
    print("File relationship graph built successfully")

if __name__ == "__main__":
    main()
