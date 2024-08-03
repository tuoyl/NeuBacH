import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


# Custom Dataset class for PyTorch
class HXMTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Define the FeatureGroupTokenizer with correct dimensions
class FeatureGroupTokenizer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureGroupTokenizer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)

# Define the BERTground model
class BERTgroundModel(nn.Module):
    def __init__(self, num_groups, group_sizes, token_dim, num_transformer_layers, num_heads, output_dim):
        super(BERTgroundModel, self).__init__()
        self.tokenizers = nn.ModuleList([FeatureGroupTokenizer(group_sizes[i], token_dim) for i in range(num_groups)])
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.fc = nn.Linear(token_dim, output_dim)

    def forward(self, x):
        tokens = []
        start = 0
        for tokenizer in self.tokenizers:
            end = start + tokenizer.fc.in_features
            tokens.append(tokenizer(x[:, start:end]))
            start = end
        tokens = torch.stack(tokens, dim=1)  # (batch_size, num_tokens, token_dim)
        transformer_output = self.transformer(tokens)  # (batch_size, num_tokens, token_dim)
        output = self.fc(transformer_output[:, 0, :])  # Use the output corresponding to the [CLS] token
        return output


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


# Evaluation function
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)

    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')



if __name__ == "__main__":

    # Load the CSV file
    df = pd.read_csv('out/hxmt_dataset.csv')

    # Split the data into features and targets
    features = df.columns[:58+256]  # First 46 columns are features
    targets = df.columns[256:]   # Remaining columns are target photon counts

    # Convert the DataFrame to numpy arrays
    X = df[features].values
    y = df[targets].values

    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


    #======== Create DataLoader objects
    train_dataset = HXMTDataset(X_train, y_train)
    val_dataset = HXMTDataset(X_val, y_val)
    test_dataset = HXMTDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    #======== Parameters for the model
    num_groups = 6
    group_sizes = [6, 3, 6, 3, 39, 256]  # These should sum to 46 (total number of features)
    token_dim = 64
    num_transformer_layers = 4
    num_heads = 8
    output_dim = len(targets)

    # Instantiate the model
    model = BERTgroundModel(num_groups, group_sizes, token_dim, num_transformer_layers, num_heads, output_dim)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    #======== Training the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=500)

    # Evaluating the model
    evaluate_model(model, test_loader, criterion)

    torch.save(model.state_dict(), 'out/bertground_model.pth')




