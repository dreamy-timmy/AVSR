# ============================================================
# 1. TOKENIZER
# ============================================================
class CharTokenizer:
    def __init__(self):
        chars = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя "
        chars += "abcdefghijklmnopqrstuvwxyz0123456789.,!?-"
        self.vocab     = ["<blank>", "<unk>"] + list(chars)
        self.char2id   = {c: i for i, c in enumerate(self.vocab)}
        self.id2char   = {i: c for i, c in enumerate(self.vocab)}

    def encode(self, text):
        return [self.char2id.get(c, 1) for c in text.lower()]

    def decode_greedy(self, ids):
        result, prev = [], -1
        for i in ids:
            if i != prev and i != 0:
                result.append(self.id2char.get(i, ""))
            prev = i
        return "".join(result)

tokenizer  = CharTokenizer()
VOCAB_SIZE = len(tokenizer.vocab)
