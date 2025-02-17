# Neural Language Modeling Assignment

## Overview

This repository contains the implementation for a Neural Language Modeling assignment. The project includes three types of language models: Feed Forward Neural Network (FFNN), Vanilla Recurrent Neural Network (RNN), and Long Short-Term Memory (LSTM). Each model is trained to perform Next Word Prediction (NWP) using the provided corpora.

### Models
1. **Feed Forward Neural Network (FFNN)**: Uses n-grams (n=3 and n=5) for fixed-size context window.
2. **Vanilla Recurrent Neural Network (RNN)**: Captures sequential dependencies with a hidden state.
3. **Long Short-Term Memory (LSTM)**: Addresses vanishing gradient issues with memory cells and gating mechanisms.

---

## File Structure

```
.
├── preprocessing.py        # Preprocesses data, generates n-grams, builds vocabulary.
├── ffnn_model.py           # Defines and trains the FFNN language model.
├── rnn_model.py            # Defines and trains the RNN language model.
├── lstm_model.py           # Defines and trains the LSTM language model.
├── perplexity.py           # Computes perplexity scores for models.
├── generator.py            # Generates predictions for the next word using the trained models.
├── processed_data.pt       # Preprocessed data (n-grams, vocabulary, etc.).
├── README.md               # Instructions and details about the project.
├── *.pt (model weights)    # Pretrained model weights.
└── report.pdf              # Analysis of results and observations.
```

---

## Requirements

- **Programming Language**: Python 3.8+
- **Framework**: PyTorch
- **Other Dependencies**:
  - `nltk`
  - `math`
  - `argparse`

### Install Required Libraries

Use the following command to install PyTorch with CUDA 11.8 support and other required dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install nltk
```

---

## Usage

### 1. Preprocessing the Data

Run the preprocessing script to generate the processed dataset:

```bash
python preprocessing.py
```

This will output `processed_data.pt`, which contains n-grams, train/test splits, and the vocabulary.

---

### 2. Training Models

#### Train FFNN
```bash
python ffnn_model.py
```

#### Train RNN
```bash
python rnn_model.py
```

#### Train LSTM
```bash
python lstm_model.py
```

Each script will save the model weights in `.pt` files (e.g., `ffnn_model_pride_and_prejudice_n3.pt`).

---

### 3. Evaluate Perplexity

Run the `perplexity.py` script to compute perplexity scores for training and test datasets:

```bash
python perplexity.py
```

The script will generate `perplexity_results_{model_type}_{corpus}.txt` files for each model and corpus.

---

### 4. Generate Predictions

Use the `generator.py` script to generate next-word predictions:

```bash
python generator.py <model_type> <corpus> <k>
```

#### Example:
```bash
python generator.py ffnn pride_and_prejudice 3
```

- **Input sentence**: `An apple a day keeps the doctor`
- **Output**:
  ```
  away: 0.4
  happy: 0.2
  fresh: 0.1
  ```

To exit the script, type `exit`.

---

## Pretrained Models

The pretrained models can be found in the `.pt` files. If not included, use the scripts above to train the models on your machine.

---

## Report

The `report.pdf` file includes:

1. Average perplexity scores for training and test datasets.
2. Model comparisons based on perplexity.
3. Analysis of model performance on longer sentences.
4. Observations on the impact of n-gram size on FFNN.

---

## Assumptions and Notes

1. **Corpus Handling**: 
   - Ensure the corpora are in `.txt` format and located in the same directory.
   - Modify file paths in the scripts if needed.
2. **Preprocessing**:
   - Vocabulary includes only words seen in the training dataset.
   - Unknown words are excluded from predictions.

---

## Execution Instructions

1. Run `preprocessing.py` to preprocess the data.
2. Train models (`ffnn_model.py`, `rnn_model.py`, `lstm_model.py`).
3. Evaluate perplexity using `perplexity.py`.
4. Use `generator.py` to predict the next word in a sequence.

---

## References

1. PyTorch Documentation
2. Lecture Slides on Neural Networks
3. Articles on RNNs and LSTMs

---

## Authors

This implementation was completed as part of a language modeling assignment.
