import sentencepiece as spm
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset, DataLoader, random_split


DATASET_FILE = "dataset.txt"
TOKENIZER_SRC_PATH = "bpe_tokenizer_en.model"
TOKENIZER_TGT_PATH = "bpe_tokenizer_hi.model"


def load_custom_dataset(filepath):
    en_sentences, hi_sentences = [], []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) == 2: 
                en_sentences.append(parts[0])
                hi_sentences.append(parts[1])
    return en_sentences, hi_sentences


def get_or_build_tokenizer(tokenizer_path, sentences):
    tokenizer_path = Path(tokenizer_path)

    if not tokenizer_path.exists():
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=8000,  
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )

        tokenizer.train_from_iterator(sentences, trainer)
        tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer trained and saved at {tokenizer_path}")
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"Loaded existing tokenizer from {tokenizer_path}")

    return tokenizer


class TranslationDataset(Dataset):
    def _init_(self, en_sentences, hi_sentences, tokenizer_src, tokenizer_tgt):
        self.en_sentences = en_sentences
        self.hi_sentences = hi_sentences
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

    def _len_(self):
        return len(self.en_sentences)

    def _getitem_(self, idx):
        en_text = self.en_sentences[idx]
        hi_text = self.hi_sentences[idx]

        en_tokens = self.tokenizer_src.encode(en_text).ids
        hi_tokens = self.tokenizer_tgt.encode(hi_text).ids

        return {
            "source": en_tokens,
            "target": hi_tokens
        }


en_sentences, hi_sentences = load_custom_dataset(DATASET_FILE)

tokenizer_src = get_or_build_tokenizer(TOKENIZER_SRC_PATH, en_sentences)
tokenizer_tgt = get_or_build_tokenizer(TOKENIZER_TGT_PATH, hi_sentences)

dataset = TranslationDataset(en_sentences, hi_sentences, tokenizer_src, tokenizer_tgt)


train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)


sample = dataset[0]
print("Tokenized Source (English):", sample["source"])
print("Tokenized Target (Hindi):", sample["target"])