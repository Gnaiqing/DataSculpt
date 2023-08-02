# LLM Aided Data Programming

## Overview
This project aims at exploring the possibility of applying large language models like GPT in the process of data programming. Especially, it asks the language models to play the "expert" role to provide label functions given specific input instances. 

## Installation
```angular2html
python -m venv .env
source .env/bin/activate
pip install transformers[sentencepiece]
pip install datasets
pip install torch torchvision torchaudio
pip install snorkel
pip install optuna
pip install wandb
pip install alipy
pip install sentence-transformers
```

## Download data (Nemo)
```angular2html
gdown 1C48r6FCw-hU6ACbO9BMIgdtHkJn6AMB2
unzip nemo_data.zip && rm nemo_data.zip
```