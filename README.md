<h1 align="center">👕 file1.agent</h1>

<h2 align="center"> Tired of main_fixed.py? file1.agent cleans up the mess from AI agents.</h2>

Tired of navigating through endless main_fixed.py and file_improved.py files? file1.agent revolutionizes your file management by eliminating AI-generated code clutter. Our advanced detection algorithms precisely **identify and remove duplicate files and mock data**, while our powerful graph visualization reveals complex file relationships at a glance. With vision model integration for comprehensive content analysis across text, images, and PDFs, file1.agent provides AI agents with complete file context.

Even better, the file relationship graph and automatically generated file summaries can **serve directly as agent memory**, giving your AI agents persistent, structured awareness of your entire workspace.

## Features

- **File Summarization**: Automatically generate summaries for files and directories
- **Duplicate Detection**: Identify duplicate files using LLM-based comparison
- **Simulated Data Detection**: Detect and remove simulated/mock data files
- **Relationship Visualization**: Create visual graphs showing file relationships
- **Vision Model Integration**: Extract and analyze content from images and PDFs
- **Reranking Support**: Use reranking models to improve relevance scoring

## Installation

You can install file1.agent using pip:

```bash
pip install file1-agent
```

For development:

```bash
git clone https://github.com/file1/file1-agent.git
cd file1-agent
pip install -e .[dev]
```

## Quick Start

Run the following commands to set up the testing environment:

```bash
mkdir -p outputs/test_repo
rm -rf outputs/test_repo
cp -r examples/test_repo outputs/
```

```python
from file1agent import FileManager
import os


file1 = FileManager(
    config="config.toml",
    analyze_dir="outputs/test_repo"
)

# Clean repository (remove duplicates and simulated data)
file1.clean_repository()
```

## Configuration

file1 uses a TOML configuration file to specify model settings and other parameters:

```toml
[llm]
language = "chs"  # Language setting: "chs" for Chinese, "eng" for English

[llm.chat] # Main language model used for general tasks and coding
model = "glm-4.5-air"  # The main language model used for general tasks
base_url = "https://cloud.infini-ai.com/maas/v1/"  # Base URL for the model API service
api_key = "<YOUR API KEY>"  # API key for accessing the language models

[llm.vision]
model = "glm-4.1v-9b-thinking"  # The vision model used for image analysis tasks
base_url = "https://open.bigmodel.cn/api/paas/v4/"  # Base URL for the vision model API service
api_key = "<YOUR API KEY>"  # API key for accessing the vision model

[rerank]
rerank_model = "bge-reranker-v2-m3"  # The reranking model used to improve search result relevance
rerank_api_key = "<YOUR API KEY>" # API key for accessing the reranking model (infiniai service)
rerank_base_url = "https://cloud.infini-ai.com/maas/v1/"  # Base URL for the reranking model API service
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Support

If you have any questions or issues, please open an issue on [GitHub Issues](https://github.com/jarrycyx/file1-agent/issues).
