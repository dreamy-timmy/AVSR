from pyctcdecode import build_ctcdecoder

from avsr.data.tokenizer import tokenizer

# tokenizer.id2char это dict {0: "<blank>", 1: "<unk>", 2: "а", ...}
# pyctcdecode хочет список где blank в позиции 0
labels = [tokenizer.id2char[i] for i in range(len(tokenizer.id2char))]

# Заменяем спец-токены на пустую строку (для blank pyctcdecode так и ожидает)
labels[0] = ""   # blank
# unk и остальные оставляем как есть

# beam_decoder = build_ctcdecoder(
#     labels=labels,
#     kenlm_model_path=None,   # пока без LM
# )
beam_decoder = build_ctcdecoder(
    labels=labels,
    kenlm_model_path="C:/kenlm_models/ru_4gram.bin",   # пока без LM
)
# print(f"Beam decoder создан, labels: {len(labels)}")


# greedy

                # preds = log_probs.argmax(dim=-1).cpu()
                # for i, pred_ids in enumerate(preds):
                #     hyp = tokenizer.decode_greedy(pred_ids.tolist())
                #     wer_scores.append(wer(texts[i], hyp))

