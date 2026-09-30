import sentencepiece as spm
from datasets import load_dataset




DATASET_NAME = "belumind/en-vi-ja-curated-500k-triplets"

SOURCE_VOCAB_SIZE = 16000
D_MODEL = 768  # Used later by the Transformer, NOT by SentencePiece.

SOURCE_CORPUS_FILE = "source_text.txt"
TOKENIZER_PREFIX = "source_tokenizer"


dataset = load_dataset(DATASET_NAME)

print(dataset)
print("Columns:", dataset["train"].column_names)


#train the tokenizer
print("\nCreating source-language corpus...")

with open(SOURCE_CORPUS_FILE, "w", encoding="utf-8") as f:
    for example in dataset["train"]:
        vietnamese = example["vi"]
        japanese = example["ja"]

        if vietnamese:
            f.write(vietnamese.strip() + "\n")

        if japanese:
            f.write(japanese.strip() + "\n")

print(f"Corpus saved to: {SOURCE_CORPUS_FILE}")



# Special tokens:
#   <pad> = 0
#   <unk> = 1
#   <bos> = 2
#   <eos> = 3
#   <VN>, <JA>, <SEP> = user-defined symbols


print("\nTraining SentencePiece tokenizer...")

spm.SentencePieceTrainer.train(
    input=SOURCE_CORPUS_FILE,
    model_prefix=TOKENIZER_PREFIX,
    vocab_size=SOURCE_VOCAB_SIZE,
    model_type="bpe",
    character_coverage=0.9995,

    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,

    user_defined_symbols=[
        "<VN>",
        "<JA>",
        "<SEP>"
    ]
)

print("\nTokenizer training complete.")
print(f"Model: {TOKENIZER_PREFIX}.model")
print(f"Vocabulary: {TOKENIZER_PREFIX}.vocab")


# ============================================================
# 4. Test the tokenizer
# ============================================================

sp = spm.SentencePieceProcessor(
    model_file=f"{TOKENIZER_PREFIX}.model"
)


def test_tokenizer(text, language):
    print("\n" + "=" * 60)
    print(f"{language} TEST")
    print("=" * 60)

    pieces = sp.encode(text, out_type=str)
    ids = sp.encode(text, out_type=int)

    print("Original:")
    print(text)

    print("\nTokens:")
    print(pieces)

    print("\nToken IDs:")
    print(ids)

    print("\nDecoded:")
    print(sp.decode(ids))


# Vietnamese example
test_tokenizer(
    "Tôi thích học máy và trí tuệ nhân tạo.",
    "Vietnamese"
)

# Japanese example
test_tokenizer(
    "私は機械学習と人工知能が好きです。",
    "Japanese"
)


# ============================================================
# 5. Check vocabulary size
# ============================================================

print("\n" + "=" * 60)
print("VOCABULARY INFORMATION")
print("=" * 60)

print("Vocabulary size:", sp.get_piece_size())

print("\nFirst 20 vocabulary pieces:")
for i in range(min(20, sp.get_piece_size())):
    print(i, repr(sp.id_to_piece(i)))



print("\nD_MODEL for the Transformer:", D_MODEL)
print("\nDone.")
