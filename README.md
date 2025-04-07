# Speech To Text Implementation using Conformer Structure

Speech2Text using Conformer structure - Training from Scratch

## Features

- Speech recognition using Vosk
- Natural language processing capabilities
- Audio processing and analysis
- Jupyter notebook support for interactive development
- Docker containerization for easy deployment

## Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chat-bot
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Installation

1. Build the Docker image:
```bash
docker build -t chat-bot .
```

2. Run the container:
```bash
./docker_run.sh
```

## Project Structure

```
.
├── audios/           # Audio files
├── configs/          # Configuration files
├── data/            # Data files
├── models/          # Model files
├── notebooks/       # Jupyter notebooks
├── output/          # Output files
├── src/             # Source code
├── tests/           # Test files
├── .devcontainer/   # VS Code devcontainer config
├── .vscode/         # VS Code settings
├── Dockerfile       # Docker configuration
├── Makefile         # Build automation
├── requirements.txt # Python dependencies
└── setup.py         # Package configuration
```

## Usage

1. Start the chat bot:
```bash
python src/main.py
```

2. For development and testing:
```bash
pytest tests/
```

## Development

- The project uses pytest for testing
- Jupyter notebooks are available in the `notebooks/` directory for interactive development
- VS Code devcontainer configuration is provided for consistent development environment

## Dependencies

- torch: Deep learning framework
- torchaudio: Audio processing
- numpy: Numerical computations
- matplotlib: Data visualization
- tqdm: Progress bars
- PyYAML: Configuration file handling
- jiwer: Speech recognition metrics
- tabulate: Table formatting
- vosk: Speech recognition
- wave: Audio file handling
- pandas: Data manipulation
- pytest: Testing framework

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

