#!/usr/bin/env python3
import os
import shutil
import sys
from pathlib import Path

# Add the parent directory to the path to import the file1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from file1agent.config import File1AgentConfig
from file1agent.file_manager import FileManager


def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_repo_dir = os.path.abspath(os.path.join(current_dir, "test_repo"))
    outputs_dir = os.path.abspath(os.path.join(current_dir, "..", "outputs"))
    config_path = os.path.abspath(os.path.join(current_dir, "..", "config.toml"))

    # Create outputs directory if it doesn't exist
    os.makedirs(outputs_dir, exist_ok=True)

    # Copy test_repo to outputs
    target_dir = os.path.join(outputs_dir, "test_repo")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(test_repo_dir, target_dir)
    print(f"Copied test_repo to {target_dir}")

    # Load configuration
    config = File1AgentConfig.from_toml(config_path)
    print(f"Loaded configuration from {config_path}")
    file_manager = FileManager(
        config=config,
        analyze_dir=target_dir,
        log_level="DEBUG",
    )

    # Test 1: Workspace Cleaning
    print("\n=== Test 1: Workspace Cleaning ===")

    # Find duplicate files
    deleted_files = file_manager.clean_repository()
    if deleted_files is None:
        print("No duplicate files found or error occurred during duplicate detection")
    else:
        print(f"Found {len(deleted_files)} duplicate pairs")

        # Print duplicate pairs
        for file_info in deleted_files:
            print(f"\nDeleted file:")
            print(f"  Path: {os.path.relpath(file_info['path'], target_dir)}")
            print(f"  Reason: {file_info['reason']}")

    print("\n=== Test 2: File Graph Building ===")
    file_manager.build_graph()
    
    print(f"\n=== Test 3: File Summary Generation ===")
    direct_summary = file_manager.search_workspace(
        max_token_cnt=1000, question="How is the data analysis performed?", use_graph=False
    )
    print(f"Generated file tree with summaries: {direct_summary}")
    
    graph_summary = file_manager.search_workspace(
        max_token_cnt=1000, question="How is the data analysis performed?", use_graph=True
    )
    print(f"Generated file graph with summaries: {graph_summary}")


    print("\n=== All Tests Completed ===")


if __name__ == "__main__":
    main()
