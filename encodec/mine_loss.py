# import torch
# import torch.nn as nn
# import torchaudio.transforms as T

# def kl_divergence_loss(z):
#     # Encourage latent space to follow N(0, 1)
#     return torch.mean(0.5 * (z**2 - torch.log(z**2 + 1e-8) - 1))

# class ReconstructionLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse_loss = nn.MSELoss()
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         device='cpu'
#         self.spectrogram = T.Spectrogram(n_fft=1024, hop_length=256, power=2).to(device)

#     def forward(self, input_waveform, reconstructed_waveform, latent):
#         # Waveform reconstruction loss
#         mse_loss = self.mse_loss(reconstructed_waveform, input_waveform)

#         # Spectrogram loss
#         input_spec = torch.log1p(self.spectrogram(input_waveform))
#         reconstructed_spec = torch.log1p(self.spectrogram(reconstructed_waveform))
#         spec_loss = self.mse_loss(reconstructed_spec, input_spec)

#         # KL divergence for latent space regularization
#         kl_loss = kl_divergence_loss(latent)

#         # Total loss
#         total_loss = mse_loss + 0.3 * spec_loss + 0.1 * kl_loss
#         return total_loss, mse_loss, spec_loss, kl_loss

import torch
import torch.nn as nn
import torchaudio.transforms as T

def kl_divergence_loss(z):
    # KL divergence for latent space regularization
    return torch.mean(0.5 * (z**2 - torch.log(z**2 + 1e-8) - 1))

class MultiScaleSpectrogramLoss(nn.Module):
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_lengths=[128, 256, 512]):
        super().__init__()
        self.spectrograms = nn.ModuleList([
            T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2)
            for n_fft, hop_length in zip(fft_sizes, hop_lengths)
        ])
        self.mse_loss = nn.MSELoss()

    def forward(self, input_waveform, reconstructed_waveform):
        loss = 0.0
        for spec in self.spectrograms:
            input_spec = torch.log1p(spec(input_waveform))
            reconstructed_spec = torch.log1p(spec(reconstructed_waveform))
            loss += self.mse_loss(reconstructed_spec, input_spec)
        return loss / len(self.spectrograms)

class ReconstructionLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.2, gamma=0.05, 
                 alpha_decay=0.98, beta_increase=0.02, gamma_increase=0.01):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.multi_spec_loss = MultiScaleSpectrogramLoss()
        
        # Dynamic loss weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.alpha_decay = alpha_decay
        self.beta_increase = beta_increase
        self.gamma_increase = gamma_increase

    def forward(self, input_waveform, reconstructed_waveform, latent):
        # Compute individual losses
        mse_loss = self.mse_loss(reconstructed_waveform, input_waveform)
        spec_loss = self.multi_spec_loss(input_waveform, reconstructed_waveform)
        kl_loss = kl_divergence_loss(latent)
        
        # Combine losses with dynamic weights
        total_loss = (self.alpha * mse_loss) + (self.beta * spec_loss) + (self.gamma * kl_loss)
        return total_loss, mse_loss, spec_loss, kl_loss
    
    def step(self):
        """Update loss weights after each epoch."""
        self.alpha *= self.alpha_decay
        self.beta += self.beta_increase
        self.gamma += self.gamma_increase

