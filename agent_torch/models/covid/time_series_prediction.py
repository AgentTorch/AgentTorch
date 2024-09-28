import warnings
import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from epiweeks import Week
import wandb

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
AGENT_TORCH_PATH = Path('/Users/shashankkumar/Documents/GitHub/AgentTorchMain/agent_torch/models/covid')
sys.path.insert(0, str(AGENT_TORCH_PATH))

from calibration.calibnn import ImprovedCovidPredictor
from calibration.utils.data import NN_INPUT_WEEKS, get_dataloader, get_labels
from calibration.utils.feature import Feature
from calibration.utils.misc import week_num_to_epiweek, name_to_neighborhood
from calibration.utils.neighborhood import Neighborhood

# Configuration
EPIWEEK_START = week_num_to_epiweek(202212)
NUM_WEEKS = 28 // 7
FEATURE_LIST = [
    Feature.RETAIL_CHANGE,
    Feature.GROCERY_CHANGE,
    Feature.PARKS_CHANGE,
    Feature.TRANSIT_CHANGE,
    Feature.WORK_CHANGE,
    Feature.RESIDENTIAL_CHANGE,
    Feature.CASES
]
LABEL_FEATURE = Feature.CASES
NEIGHBORHOOD = name_to_neighborhood("Astoria")

def parse_args():
    parser = argparse.ArgumentParser(description="AgentTorch: million-scale, differentiable agent-based models")
    parser.add_argument("-c", "--config", default="config_opt_llm.yaml", help="Name of the yaml config file with the parameters.")
    return parser.parse_args()

def normalize_data(data):
    return (data - data.mean()) / data.std()

def train_model(model, dataloader, optimizer, loss_function, device):
    model.train()
    total_loss = 0
    for episode in range(10):
        print(f"\nRunning episode {episode}...")
        optimizer.zero_grad()
        for metadata, features in dataloader:
            r0_values = model(features, metadata)
            target_weekly_cases = get_labels(NEIGHBORHOOD, EPIWEEK_START + episode, 4, LABEL_FEATURE).to(device)
            target_weekly_cases = normalize_data(target_weekly_cases)
            loss = loss_function(r0_values, target_weekly_cases)
            loss.backward()
            print(f"Predicted cases: {r0_values}, Actual cases: {target_weekly_cases}, Loss: {loss.item()}")
            optimizer.step()
            torch.cuda.empty_cache()
            total_loss += loss.item()
            
            # Log metrics to Wandb
            wandb.log({
                "episode": episode,
                "loss": loss.item(),
                "predicted_cases": r0_values.detach().cpu().numpy(),
                "actual_cases": target_weekly_cases.detach().cpu().numpy()
            })
    
    return total_loss / len(dataloader)

def main():
    args = parse_args()
    config_file = Path("/Users/shashankkumar/Documents/GitHub/MacroEcon/models/covid/yamls/config_opt_llm.yaml")
    print(f"Running experiment with config file: {config_file}")

    # Initialize Wandb
    wandb.init(entity="blankpoint",project="covid-predictor", config={
        "learning_rate": 0.001,
        "architecture": "ImprovedCovidPredictor",
        "dataset": "COVID-19",
        "epochs": 1000,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCovidPredictor(input_dim=len(FEATURE_LIST), meta_dim=len(Neighborhood)).to(device)
    loss_function = nn.HuberLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Watch the model in Wandb
    wandb.watch(model, log_freq=100)

    num_epochs = 1000
    for epoch in range(num_epochs):
        dataloader = get_dataloader(NEIGHBORHOOD, EPIWEEK_START, NUM_WEEKS, FEATURE_LIST)
        avg_loss = train_model(model, dataloader, optimizer, loss_function, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Log epoch-level metrics to Wandb
        wandb.log({
            "epoch": epoch,
            "average_loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        scheduler.step(avg_loss)

    # Close Wandb run
    wandb.finish()

if __name__ == "__main__":
    main()