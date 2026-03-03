from torchvision import transforms
import re
from collections import Counter

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def tokenize(text):
    return text.split()

def build_vocab(questions, min_freq=1):
    counter = Counter()

    for q in questions:
        q = clean_text(q)
        tokens = tokenize(q)
        counter.update(tokens)

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab

def numericalize(text, vocab):
    text = clean_text(text)
    tokens = tokenize(text)
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]


def pad_sequence(seq, max_length, pad_idx=0):
    if len(seq) < max_length:
        seq = seq + [pad_idx] * (max_length - len(seq))
    else:
        seq = seq[:max_length]
    return seq

def encode_answer(answer):
    answer = answer.lower().strip()
    if answer == "yes":
        return 1.0
    elif answer == "no":
        return 0.0
    else:
        raise ValueError(f"Unexpected answer: {answer}")