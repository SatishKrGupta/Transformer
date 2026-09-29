
import re


class Tokenizer:
    def __init__(self):
        # Special tokens used by our Transformer
        self.special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>"
        ]

        self.token_to_id = {}
        self.id_to_token = {}

    
    def tokenize(self, text):
        # Split on whitespace, preserving complete Hindi words.
        return text.lower().split()

    def build_vocab(self, sentences):
        # Collect all unique tokens.
        vocab = set()

        for sentence in sentences:
            tokens = self.tokenize(sentence)
            vocab.update(tokens)

        # Special tokens receive the first four IDs.
        all_tokens = self.special_tokens + sorted(
            vocab - set(self.special_tokens)
        )

        self.token_to_id = {
            token: idx
            for idx, token in enumerate(all_tokens)
        }

        # Reverse mapping: ID -> token
        self.id_to_token = {
            idx: token
            for token, idx in self.token_to_id.items()
        }

    def encode(self, text):
        tokens = self.tokenize(text)

        # Add beginning and end markers.
        tokens = ["<BOS>"] + tokens + ["<EOS>"]

        # Unknown words get the <UNK> ID.
        return [
            self.token_to_id.get(
                token,
                self.token_to_id["<UNK>"]
            )
            for token in tokens
        ]

    def decode(self, token_ids):
        tokens = [
            self.id_to_token.get(idx, "<UNK>")
            for idx in token_ids
        ]

        # Remove special tokens from the output.
        tokens = [
            token for token in tokens
            if token not in self.special_tokens
        ]

        text = " ".join(tokens)

        # Remove spaces before punctuation.
        text = re.sub(r"\s+([.,!?।])", r"\1", text)

        return text





# Our tiny bilingual example
english_sentences = [
    "I love India",
    "How are you?",
    "I am learning AI"
]

hindi_sentences = [
    "मुझे भारत से प्यार है।",
    "आप कैसे हैं?",
    "मैं एआई सीख रहा हूँ।"
]

# Build one shared vocabulary for both languages.
tokenizer = Tokenizer()

tokenizer.build_vocab(
    english_sentences + hindi_sentences
)

# Test English
english = "I love India"
english_ids = tokenizer.encode(english)

print("English:", english)
print("Tokens:", tokenizer.tokenize(english))
print("Token IDs:", english_ids)
print("Decoded:", tokenizer.decode(english_ids))

# Test Hindi
hindi = "मुझे भारत से प्यार है।"
hindi_ids = tokenizer.encode(hindi)

print("\nHindi:", hindi)
print("Tokens:", tokenizer.tokenize(hindi))
print("Token IDs:", hindi_ids)
print("Decoded:", tokenizer.decode(hindi_ids))

print("\nVocabulary size:", len(tokenizer.token_to_id))