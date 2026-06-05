import torch

def seconds_to_timestamp(sec):
    """123.4 → '00:02:03'"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

