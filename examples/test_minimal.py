from file1agent import FileManager
import os


file1 = FileManager(
    config="config.toml",
    analyze_dir="outputs/test_repo"
)

# Clean repository (remove duplicates and simulated data)
file1.clean_repository()