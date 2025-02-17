import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class FFNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_gram):
        super(FFNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear((n_gram - 1) * embed_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

def load_data(corpus_key):
    data = torch.load("processed_data.pt")[corpus_key]
    train_data = torch.tensor(data["train"], dtype=torch.long)  # Ensure correct data type
    vocab_size = len(data["vocab"]) + 1
    inputs, targets = train_data[:, :-1], train_data[:, -1]
    return inputs, targets, vocab_size

def train_ffnn(n_gram_values=[3, 5], corpora=["ulysses"]):
    for corpus in corpora:
        for n_gram in n_gram_values:
            corpus_key = f"{corpus}_n{n_gram}"
            inputs, targets, vocab_size = load_data(corpus_key)

            dataset = TensorDataset(inputs, targets)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

            model = FFNNLanguageModel(vocab_size, embed_size=128, hidden_size=256, n_gram=n_gram).to(device)
            criterion = nn.NLLLoss()
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

            torch.save(model.state_dict(), f"ffnn_model_{corpus}_n{n_gram}.pt")
            print(f"Model saved as ffnn_model_{corpus}_n{n_gram}.pt")

if __name__ == "__main__":
    train_ffnn(n_gram_values=[3, 5], corpora=["ulysses"])
