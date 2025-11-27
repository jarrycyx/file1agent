#!/usr/bin/env python3
import os
import shutil
import sys
from pathlib import Path

# Add the parent directory to the path to import the file1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from file1.config import File1Config
from file1.file_manager import FileManager
from file1.file_summary import FileSummary

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_repo_dir = os.path.join(current_dir, "test_repo")
    outputs_dir = os.path.join(current_dir, "..", "outputs")
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
    
    # Test 1: File Summary Generation
    print("\n=== Test 1: File Summary Generation ===")
    file_summary = FileSummary(
        config=config,
        analyze_dir=target_dir,
        summary_cache_path=os.path.join(outputs_dir, "file_summary_cache.json")
    )
    
    # Get file tree with summaries
    file_tree_str = file_summary.get_file_tree_with_summaries()
    print(f"Generated file tree with summaries")
    
    # Print file tree with summaries
    print(file_tree_str)
    
    # Print summaries for each file from cache
    print(f"\nFile summaries from cache:")
    for file_path, file_info in file_summary.file_cache.items():
        print(f"\nFile: {os.path.relpath(file_path, target_dir)}")
        print(f"Summary: {file_info['summary'][:200]}..." if len(file_info['summary']) > 200 else f"Summary: {file_info['summary']}")
    
    # Test 2: File Deduplication
    print("\n=== Test 2: File Deduplication ===")
    file_manager = FileManager(
        config=config,
        analyze_dir=target_dir,
        summary_cache_path=os.path.join(outputs_dir, "file_summary_cache.json"),
        backup_path=os.path.join(outputs_dir, "backup")
    )
    
    # Get file summaries
    summaries = file_manager._get_file_summaries()
    print(f"Found {len(summaries)} files to analyze for duplicates")
    
    # Find duplicate files
    duplicates = file_manager._find_duplicates_with_reranker(summaries)
    if duplicates is None:
        print("No duplicate files found or error occurred during duplicate detection")
    else:
        print(f"Found {len(duplicates)} duplicate pairs")
        
        # Print duplicate pairs
        for pair in duplicates:
            print(f"\nDuplicate pair:")
            print(f"  File 1: {os.path.relpath(pair[0], target_dir)}")
            print(f"  File 2: {os.path.relpath(pair[1], target_dir)}")
            print(f"  Similarity: {pair[2]:.2f}")
    
    # Test 3: Simulated Data Detection
    print("\n=== Test 3: Simulated Data Detection ===")
    # Check each code file for simulated data
    code_extensions = {".py", ".sh", ".c", ".cpp", ".r"}
    simulated_files = []
    
    for file_path in summaries.keys():
        _, ext = os.path.splitext(file_path)
        if ext.lower() in code_extensions:
            is_simulated = file_manager._detect_simulated_data(file_path)
            if is_simulated:
                simulated_files.append(file_path)
                print(f"Detected simulated data in: {os.path.relpath(file_path, target_dir)}")
    
    print(f"Found {len(simulated_files)} files with simulated data")
    
    # Test 4: Full Repository Cleaning
    print("\n=== Test 4: Full Repository Cleaning ===")
    # Reset the test directory by copying it again
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(test_repo_dir, target_dir)
    
    # Create a new file manager instance
    file_manager_clean = FileManager(
        config=config,
        analyze_dir=target_dir,
        summary_cache_path=os.path.join(outputs_dir, "file_summary_cache_clean.json"),
        backup_path=os.path.join(outputs_dir, "backup_clean")
    )
    
    # Clean the repository
    deleted_files = file_manager_clean.clean_repository()
    print(f"Repository cleaning completed. Deleted {len(deleted_files)} files")
    
    # Print details about deleted files
    for file_info in deleted_files:
        print(f"Deleted: {os.path.relpath(file_info['path'], target_dir)} - Reason: {file_info['reason']}")
    
    # Print remaining files
    remaining_files = []
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            remaining_files.append(os.path.relpath(os.path.join(root, file), target_dir))
    
    print(f"\nRemaining files after cleaning: {len(remaining_files)}")
    for file in remaining_files:
        print(f"  - {file}")
    
    print("\n=== Test 5: File Graph Building ===")
    file_manager_clean.build_graph()
    
    
    print("\n=== All Tests Completed ===")

if __name__ == "__main__":
    main()