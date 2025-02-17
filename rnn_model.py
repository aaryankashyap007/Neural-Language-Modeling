import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, hidden = self.rnn(x)
        x = self.fc(output[:, -1, :])  # Get last time-step output
        return x

def load_data(corpus_key):
    data = torch.load("processed_data.pt")[corpus_key]
    train_data = torch.tensor(data["train"], dtype=torch.long, device=device)  # Ensure data is on GPU
    vocab_size = len(data["vocab"]) + 1
    inputs, targets = train_data[:, :-1], train_data[:, -1]
    return inputs, targets, vocab_size

def train_rnn(n_gram_values=[3, 5], corpora=["ulysses"]):
    for corpus in corpora:
        for n_gram in n_gram_values:
            corpus_key = f"{corpus}_n{n_gram}"
            inputs, targets, vocab_size = load_data(corpus_key)

            dataset = TensorDataset(inputs, targets)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

            model = RNNLanguageModel(vocab_size, embed_size=128, hidden_size=256).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(5):
                for batch_inputs, batch_targets in dataloader:
                    batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_inputs)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                
                print(f"{corpus}, n={n_gram}, Epoch {epoch+1}: Loss = {loss.item()}")

            torch.save(model.state_dict(), f"rnn_model_{corpus}_n{n_gram}.pt")
            print(f"Model saved as rnn_model_{corpus}_n{n_gram}.pt")

if __name__ == "__main__":
    train_rnn(n_gram_values=[3, 5], corpora=["ulysses"])
