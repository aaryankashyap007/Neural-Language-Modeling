import torch
import torch.nn as nn
import math
from ffnn_model import FFNNLanguageModel
from rnn_model import RNNLanguageModel
from lstm_model import LSTMLanguageModel

# Set up CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model(model_type, corpus, n_gram):
    vocab = torch.load("processed_data.pt")[f"{corpus}_n{n_gram}"]["vocab"]
    vocab_size = len(vocab) + 1

    if model_type == "ffnn":
        model = FFNNLanguageModel(vocab_size, embed_size=128, hidden_size=256, n_gram=n_gram)
    elif model_type == "rnn":
        model = RNNLanguageModel(vocab_size, embed_size=128, hidden_size=256)
    elif model_type == "lstm":
        model = LSTMLanguageModel(vocab_size, embed_size=128, hidden_size=256, num_layers=2)
    else:
        raise ValueError("Invalid model type. Choose from: ffnn, rnn, lstm.")

    model.load_state_dict(torch.load(f"{model_type}_model_{corpus}_n{n_gram}.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, vocab

def compute_sentence_perplexity(model, sentence, n_gram):
    total_log_prob = 0
    total_words = 0

    for i in range(len(sentence) - (n_gram - 1)):  
        input_seq = torch.tensor([sentence[i:i+(n_gram-1)]], dtype=torch.long, device=device)
        target = torch.tensor([sentence[i+(n_gram-1)]], dtype=torch.long, device=device)

        with torch.no_grad():
            output = model(input_seq)  # [1, vocab_size]
            log_prob = -output.log_softmax(dim=1).gather(1, target.unsqueeze(1)).sum().item()
        
        total_log_prob += log_prob
        total_words += 1

    if total_words == 0:
        return float('inf')

    avg_log_prob = total_log_prob / total_words
    return math.exp(avg_log_prob)

def compute_dataset_perplexity(model, dataset, n_gram):
    sentence_perplexities = [compute_sentence_perplexity(model, sentence, n_gram) for sentence in dataset]
    avg_perplexity = sum(sentence_perplexities) / len(sentence_perplexities)
    return sentence_perplexities, avg_perplexity

if __name__ == "__main__":
    n_gram = 3
    model_types = ['ffnn', 'rnn', 'lstm']
    corpuses = ['pride_and_prejudice', 'ulysses']

    for model_type in model_types:
        for corpus in corpuses:
            print(f"Evaluating {model_type.upper()} on {corpus}...")

            model, vocab = load_model(model_type, corpus, n_gram)
            data = torch.load("processed_data.pt")[f"{corpus}_n{n_gram}"]
            
            train_perplexities, avg_train_perplexity = compute_dataset_perplexity(model, data["train"], n_gram)
            test_perplexities, avg_test_perplexity = compute_dataset_perplexity(model, data["test"], n_gram)

            with open(f"perplexity_results_{model_type}_{corpus}.txt", "w") as f:
                f.write(f"Perplexity Results for {model_type.upper()} on {corpus}\n")
                f.write("--------------------------------------------------------\n\n")
                f.write(f"Average Training Perplexity: {avg_train_perplexity:.4f}\n")
                f.write(f"Average Test Perplexity: {avg_test_perplexity:.4f}\n")
