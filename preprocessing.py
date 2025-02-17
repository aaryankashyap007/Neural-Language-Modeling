import re
import random
import torch
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from collections import Counter

class TextProcessor:
    def __init__(self, file_path, n_gram=3):
        self.file_path = file_path
        self.n_gram = n_gram
        self.vocab = {}
        self.word_to_index = {}
        self.index_to_word = {}

    def load_sentences(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            text = file.read()
        text = re.sub(r'\n(?=\S)', ' ', text)
        text = re.sub(r'-+', ' ', text)
        text = text.replace('_', '')
        return sent_tokenize(text)

    def tokenize_words(self, sentences):
        tokenizer = RegexpTokenizer(r"\w+|[^\w\s]+")
        return [tokenizer.tokenize(sentence) for sentence in sentences]

    def build_vocab(self, tokenized_sentences):
        words = [word for sentence in tokenized_sentences for word in sentence]
        word_counts = Counter(words)
        self.vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), 1)}
        self.word_to_index = self.vocab
        self.index_to_word = {idx: word for word, idx in self.vocab.items()}

    def generate_ngrams(self, tokenized_sentences):
        sequences = []
        for sentence in tokenized_sentences:
            if len(sentence) >= self.n_gram:
                sequences.extend([sentence[i : i + self.n_gram] for i in range(len(sentence) - self.n_gram + 1)])
        return sequences

    def process(self):
        sentences = self.load_sentences()
        random.shuffle(sentences)
        test_sentences = sentences[:1000]
        train_sentences = sentences[1000:]

        test_tokens = self.tokenize_words(test_sentences)
        train_tokens = self.tokenize_words(train_sentences)

        self.build_vocab(train_tokens + test_tokens)

        train_sequences = self.generate_ngrams(train_tokens)
        test_sequences = self.generate_ngrams(test_tokens)

        train_sequences = [[self.word_to_index[word] for word in seq] for seq in train_sequences]
        test_sequences = [[self.word_to_index[word] for word in seq] for seq in test_sequences]

        return train_sequences, test_sequences, self.word_to_index, self.index_to_word

if __name__ == "__main__":
    n_values = [3, 5]
    corpora = ["pride_and_prejudice.txt", "ulysses.txt"]
    processed_data = {}

    for n in n_values:
        for corpus in corpora:
            processor = TextProcessor(corpus, n_gram=n)
            train_set, test_set, word_to_index, index_to_word = processor.process()
            key = f"{corpus.split('.')[0]}_n{n}"
            processed_data[key] = {"train": train_set, "test": test_set, "vocab": word_to_index}

    torch.save(processed_data, "processed_data.pt")
