import torch
import torch.nn as nn
import torch.optim as optim

# ---- 1. Define the neural network ----
class RamanPINN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # output = predicted concentration
        )
        # Beer–Lambert proportionality constant: I_peak ≈ k * concentration
        self.k = nn.Parameter(torch.tensor(1.0))  # learnable scalar

    def forward(self, x):
        # x shape: (batch_size, input_dim) – Raman spectra
        conc_pred = self.net(x)          # predicted concentration
        return conc_pred



model = RamanPINN(input_dim=spectra.shape[1])
optimizer = optim.Adam(model.parameters(), lr=1e-3)

lambda_beer = 0.1  # weight for physics term (you can tune this)

criterion = nn.MSELoss()

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # ---- Forward ----
    conc_pred = model(spectra)              # (batch_size, 1)

    # 1) Data loss: prediction vs true labels
    data_loss = criterion(conc_pred, true_conc)

    # 2) Physics loss: Beer–Lambert I_peak ≈ k * concentration
    # extract peak intensity from each spectrum
    I_peak = spectra[:, peak_index].unsqueeze(1)  # (batch_size, 1)

    # predicted intensity from Beer–Lambert
    I_pred_physics = model.k * conc_pred          # (batch_size, 1)

    beer_lambert_loss = criterion(I_peak, I_pred_physics)

    # ---- Total loss = data + physics ----
    total_loss = data_loss + lambda_beer * beer_lambert_loss

    # ---- Backprop & update ----
    total_loss.backward()
    optimizer.step()

    # (optional) print progress
    # print(epoch, total_loss.item(), data_loss.item(), beer_lambert_loss.item())
