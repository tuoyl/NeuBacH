import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
## --------------- Evaluate
import matplotlib.pyplot as plt


validate_data = pd.read_csv("./out/P010129300102.csv")

from train_nicer import BERTgroundModel
# Instantiate the model
#======== Parameters for the model
num_groups = 6
group_sizes = [6, 3, 6, 3, 39, 256]  # These should sum to 46 (total number of features)
token_dim = 64
num_transformer_layers = 4
num_heads = 8
output_dim = 256

model = BERTgroundModel(num_groups, group_sizes, token_dim, num_transformer_layers, num_heads, output_dim)

# Load the state dictionary
model.load_state_dict(torch.load('./out/bertground_model.pth',  map_location=torch.device('cpu')))

# Set the model to evaluation mode
model = model.to('cpu')
model.eval()


# Convert to DataFrame and then to tensor
validate_tensor = torch.tensor(validate_data.values, dtype=torch.float32)

# Step 2: Use the Trained Model to Make Predictions
with torch.no_grad():
    predicted_spectrum = model(validate_tensor).numpy()

print(predicted_spectrum)
## Generate fake true spectrum for comparison (randomly generated)
#true_spectrum = np.random.poisson(5, output_dim)
#
## Step 3: Plot the True and Predicted Spectra
#energy_bins = np.arange(0.2, 0.2 + 0.01 * output_dim, 0.01)
#
#plt.style.use(['nature'])
#import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
#
#plt.subplot(211)
#plt.plot(energy_bins, true_spectrum, label='True Spectrum', color='blue', linestyle='--')
#plt.plot(energy_bins, predicted_spectrum.flatten(), label='Predicted Spectrum', color='red')
#plt.subplot(212)
#plt.errorbar(energy_bins, true_spectrum - predicted_spectrum.flatten(), fmt='.')
#plt.xlabel('Energy (keV)')
#plt.ylabel('Photon Count')
#plt.title('True vs Predicted Spectrum')
#plt.legend()
#plt.grid(True)
#plt.show()
