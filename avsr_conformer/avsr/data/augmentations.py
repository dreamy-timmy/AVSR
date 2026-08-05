import torch
import torch.nn as nn

class SpecAugment(nn.Module):
    """ 
    Регуляризация — модель учится восстанавливать речь при потере части информации.
    
    """
    def __init__(self,
                 time_mask_width=40,    # макс ширина маски по времени (в фреймах)
                 freq_mask_width=15,    # макс ширина маски по частоте (мел-фильтрам)
                 n_time_masks=2,        # сколько масок по времени
                 n_freq_masks=2):       # сколько масок по частоте
        super().__init__()
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        self.n_time_masks    = n_time_masks
        self.n_freq_masks    = n_freq_masks
    
    def forward(self, x):
        """
        x: (B, T, F) — мел-спектрограмма
        """
        if not self.training:
            return x   # при инференсе ничего не делаем
        
        x = x.clone()
        B, T, F = x.shape
        
        for b in range(B):
            # Маски по времени
            for _ in range(self.n_time_masks):
                t = torch.randint(0, self.time_mask_width + 1, (1,)).item()
                if t == 0 or T - t <= 0:
                    continue
                t0 = torch.randint(0, T - t, (1,)).item()
                x[b, t0:t0+t, :] = 0
            
            # Маски по частоте
            for _ in range(self.n_freq_masks):
                f = torch.randint(0, self.freq_mask_width + 1, (1,)).item()
                if f == 0 or F - f <= 0:
                    continue
                f0 = torch.randint(0, F - f, (1,)).item()
                x[b, :, f0:f0+f] = 0
        
        return x

