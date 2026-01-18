import json
import regex as re
from collections import defaultdict, Counter
import pickle


class CustomTokenizer:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_freq = Counter()
        self.bpe_merges = []
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }

        for token, idx in self.special_tokens.items():
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token


    def train(self, texts, min_freq=2):
        """train the tokenizer on corpus"""
        word_counts = Counter()

        for text in texts:
            words = self._preprocess_text(text)
            word_counts.update(words)

        char_vocab = set()
        for word in word_counts:
            char_vocab.update(list(word))

        current_vocab = {char: idx + len(self.special_tokens)
                        for idx, char in enumerate(sorted(char_vocab))}

        for token, idx in current_vocab.items():
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

        word_splits = {}
        for word in word_counts:
            word_splits[word] = list(word)

        num_merges = self.vocab_size - len(current_vocab) - len(self.special_tokens)

        for i in range(num_merges):
            pairs = defaultdict(int)

            for word, freq in word_counts.items():
                symbols = word_splits[word]
                for j in range(len(symbols)-1):
                    pairs[(symbols[j], symbols[j+1])] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)

            self.bpe_merges.append(best_pair)

            new_token = ''.join(best_pair)
            new_idx = len(self.word_to_id)
            self.word_to_id[new_token] = new_idx
            self.id_to_word[new_idx] = new_token

            for word in word_splits:
                symbols = word_splits[word]
                new_symbols = []
                j = 0
                while j < len(symbols):
                    if j < len(symbols) - 1 and (symbols[j], symbols[j+1]) == best_pair:
                        new_symbols.append(new_token)
                        j += 2
                    else:
                        new_symbols.append(symbols[j])
                        j += 1
                word_splits[word] = new_symbols


    def _preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        words = text.split()
        return words


    def encode(self, text):
        """convert text to token ids"""
        tokens = []
        tokens.append(self.special_tokens['<BOS>'])

        words = self._preprocess_text(text)

        for word in words:
            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                if token in self.word_to_id:
                    tokens.append(self.word_to_id[token])
                else:
                    tokens.append(self.special_tokens['<UNK>'])

        tokens.append(self.special_tokens['<EOS>'])
        return tokens


    def _tokenize_word(self, word):
        if not word:
            return []

        symbols = list(word)

        for merge in self.bpe_merges:
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == merge:
                    new_symbols.append(''.join(merge))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols


    def decode(self, token_ids):
        """convert token ids back to text"""
        tokens = []
        for idx in token_ids:
            if idx in self.id_to_word:
                token = self.id_to_word[idx]
                if token not in self.special_tokens:
                    tokens.append(token)

        text = ''.join(tokens)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        return text


    def save(self, filepath):
        data = {
            'vocab_size': self.vocab_size,
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'bpe_merges': self.bpe_merges,
            'special_tokens': self.special_tokens
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)


    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.vocab_size = data['vocab_size']
        self.word_to_id = data['word_to_id']
        self.id_to_word = data['id_to_word']
        self.bpe_merges = data['bpe_merges']
        self.special_tokens = data['special_tokens']
