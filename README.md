# file1: Tired of main_fixed.py? file1 cleans up the mess from AI agents. / 告别main_fixed.py：用file1清理AI智能体留下的烂摊子

Tired of navigating through endless main_fixed.py and file_improved.py files? File1 revolutionizes your file management by eliminating AI-generated code clutter. Our advanced detection algorithms precisely **identify and remove duplicate files and mock data**, while our powerful graph visualization reveals complex file relationships at a glance. With vision model integration for comprehensive content analysis across text, images, and PDFs, File1 provides AI agents with complete file context.

Even better, the file relationship graph and automatically generated file summaries can **serve directly as agent memory**, giving your AI agents persistent, structured awareness of your entire workspace.
Turn your chaotic codebase into an organized, intelligent environment with File1.

## Features

- **File Summarization**: Automatically generate summaries for files and directories
- **Duplicate Detection**: Identify duplicate files using LLM-based comparison
- **Simulated Data Detection**: Detect and remove simulated/mock data files
- **Relationship Visualization**: Create visual graphs showing file relationships
- **Vision Model Integration**: Extract and analyze content from images and PDFs
- **Reranking Support**: Use reranking models to improve relevance scoring

## Installation

You can install File1 using pip:

```bash
pip install file1
```

For development:

```bash
git clone https://github.com/file1/file1.git
cd file1
pip install -e .[dev]
```

## Quick Start

```python
from file1 import File1
from file1.config import File1Config

# Initialize with default configuration
config = File1Config()
file1 = File1(config)

# Clean repository (remove duplicates and simulated data)
file1.clean_repository("/path/to/your/project")

# Build file relationship graph
graph = file1.build_graph("/path/to/your/project")

# Visualize the graph
file1.visualize_graph(graph, save_path="file_relationship.png")
```

## Configuration

File1 uses a TOML configuration file to specify model settings and other parameters:

```toml
[model]
name = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"
api_key = "your-api-key"

[llm.chat]
model = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"
api_key = "your-api-key"

[llm.vision]
model = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"
api_key = "your-api-key"

[reranker]
model = "bge-reranker-v2-m3"
base_url = "https://api.bge-m3.com/v1"
api_key = "your-reranker-api-key"

[save_path]
path = "/path/to/save/directory"
```

## API Reference

### File1

The main class for file analysis and management.

#### Methods

- `clean_repository(directory)`: Remove duplicate files and simulated data
- `build_graph(directory)`: Build a file relationship graph
- `visualize_graph(graph, save_path)`: Visualize the file relationship graph

### FileSummary

A class for generating file and directory summaries.

#### Methods

- `summarize_file(file_path)`: Generate a summary for a single file
- `summarize_directory(directory_path)`: Generate a summary for a directory
- `get_file_tree_with_summaries(directory_path)`: Get a file tree with summaries

### FileManager

A class for managing files, detecting duplicates, and building relationships.

#### Methods

- `detect_file_duplication(file1, file2)`: Check if two files are duplicates
- `detect_simulated_data(file_path)`: Check if a file contains simulated data
- `find_duplicates_with_reranker(files)`: Find duplicates using reranking

## Examples

### Basic Usage

```python
from file1 import File1
from file1.config import File1Config

# Initialize with configuration
config = File1Config.from_file("config.toml")
file1 = File1(config)

# Analyze a directory
summary = file1.summarize_directory("/path/to/project")
print(summary)
```

### Quick Test Example

```python
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
```

### Custom Configuration

```python
from file1.config import ModelConfig, LLMConfig, RerankConfig, File1Config

# Create custom configuration
model_config = ModelConfig(
    name="gpt-4o",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key"
)

llm_config = LLMConfig(
    chat=model_config,
    vision=model_config
)

rerank_config = RerankConfig(
    model="bge-reranker-v2-m3",
    base_url="https://api.bge-m3.com/v1",
    api_key="your-reranker-api-key"
)

config = File1Config(
    llm=llm_config,
    reranker=rerank_config,
    save_path="/path/to/save/directory"
)

file1 = File1(config)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Support

If you have any questions or issues, please open an issue on [GitHub Issues](https://github.com/file1/file1/issues).