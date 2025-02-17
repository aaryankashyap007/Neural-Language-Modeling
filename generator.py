import torch
import torch.nn as nn
import argparse
import re
from preprocessing import TextProcessor
from ffnn_model import FFNNLanguageModel
from rnn_model import RNNLanguageModel
from lstm_model import LSTMLanguageModel

def load_model(model_type, corpus):
    n_gram = 3
    vocab = torch.load("processed_data.pt")[f"{corpus}_n{n_gram}"]["vocab"]
    vocab_size = len(vocab) + 1

    if model_type == "ffnn":
        model = FFNNLanguageModel(vocab_size, embed_size=128, hidden_size=256, n_gram=n_gram)
    elif model_type == "rnn":
        model = RNNLanguageModel(vocab_size, embed_size=128, hidden_size=256)
    elif model_type == "lstm":
        model = LSTMLanguageModel(vocab_size, embed_size=128, hidden_size=256)
    else:
        raise ValueError("Invalid model type. Choose from: ffnn, rnn, lstm.")

    model.load_state_dict(torch.load(f"{model_type}_model_{corpus}_n{n_gram}.pt", map_location=torch.device('cpu')))
    model.eval()
    return model, vocab

def preprocess_input(sentence, vocab):
    n_gram = 3
    words = re.sub(r"[^a-zA-Z\s]", "", sentence.lower()).split()
    words = words[-(n_gram-1):]
    input_seq = [vocab[word] for word in words if word in vocab]

    while len(input_seq) < (n_gram - 1):
        input_seq.insert(0, 0)

    return torch.tensor([input_seq], dtype=torch.long)

def predict_next_word(model, vocab, input_seq, k):
    with torch.no_grad():
        output = model(input_seq)
        probs = torch.softmax(output, dim=1)
        
    # Get all possible words sorted by probability
    top_k = torch.topk(probs, probs.size(1))  # Get all words sorted by their probability
    
    index_to_word = {idx: word for word, idx in vocab.items()}
    
    # Initialize the list to store predictions
    all_predictions = []
    
    for idx, prob in zip(top_k.indices[0], top_k.values[0]):
        idx_item = idx.item()
        # Ensure the index exists in the vocab before accessing it
        if idx_item in index_to_word:
            all_predictions.append((index_to_word[idx_item], prob.item()))
        else:
            # If the index is not found in the vocab, we skip it (or handle as needed)
            continue

    # Filter out only alphanumeric words
    alphanumeric_predictions = [(word, prob) for word, prob in all_predictions if word.isalnum()]

    # Sort predictions based on the probability and return top k
    sorted_predictions = sorted(alphanumeric_predictions, key=lambda x: x[1], reverse=True)[:k]

    return sorted_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", choices=["ffnn", "rnn", "lstm"])
    parser.add_argument("corpus", choices=["pride_and_prejudice", "ulysses"])
    parser.add_argument("k", type=int, help="Number of top predictions to return")
    args = parser.parse_args()

    model, vocab = load_model(args.model_type, args.corpus)

    while True:
        sentence = input("Input sentence: ")
        if sentence.lower() == 'exit':
            break
        input_seq = preprocess_input(sentence, vocab)
        predictions = predict_next_word(model, vocab, input_seq, args.k)

        print("Predicted next words:")
        for word, prob in predictions:
            print(f"{word}: {prob:.4f}")
