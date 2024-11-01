# Python Kaggle Dataset Download Instructions

## Overview
This guide provides instructions on how to download a specific dataset from Kaggle using a Python script.

## Prerequisites
- Python installed on your system.
- Kaggle API credentials set up on your machine. You can follow the [official Kaggle API instructions](https://www.kaggle.com/docs/api) for setting up your API credentials.

## Installation
Before running the script, you need to install the required Python package. You can do this by running the following command in your terminal:

```bash
pip install kagglehub
```
To download the database run the next python code:
```python
import kagglehub

# Download latest version of the dataset
path = kagglehub.dataset_download("imsparsh/lakh-midi-clean")

print("Path to dataset files:", path)
```

# Python MIDI Analyzer Usage Instructions

## Overview
This document outlines the usage of the `analisis.py` script, which is designed to interact with and analyze MIDI files stored in a specific dataset directory.

## Prerequisites
- Python installed on your system.
- Necessary Python libraries installed.
- Dataset containing MIDI files downloaded and accessible.
- Kaggle API credentials set up (for downloading the dataset).

## Installation
Before running the script, you need to install the required Python packages. Install them using the following command in your terminal:

```bash
pip install -r requirements.txt
```
## Runing
Before running the code you need to change the path to the database in the def main()

```bash
Python3 analisis.py
```

