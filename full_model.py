import torch
from models.FFNN import FeedForwardNN

# create architecture
model = FeedForwardNN(
    n_classes=2,
    n_features=500,      
    hidden_size=10,
    layers=2
)

# load weights
model.load_state_dict(torch.load("weights_only.pth", map_location="cpu"))

# save full model
torch.save(model, "full_model.pth")