import numpy as np

# ============================================================
# 5. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
def wer(reference, hypothesis):
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    d   = np.zeros((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1): d[i][0] = i
    for j in range(len(hyp) + 1): d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost   = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(ref)][len(hyp)] / max(len(ref), 1)

