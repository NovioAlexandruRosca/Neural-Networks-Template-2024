import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import wandb

# All configs remain exactly the same...
sweep_config = {
    "method": "random",
    "metric": {"name": "test_accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 0.0001, "max": 0.01},
        "hidden_size_1": {"values": [1024]},
        "hidden_size_2": {"values": [512]},
        "hidden_size_3": {"values": [256]},
        "dropout_rate": {"min": 0.1, "max": 0.5},
        "batch_size": {"values": [32, 64, 128]},
    },
}


# MLP class remains exactly the same...
class MLP(nn.Module):
    def __init__(self, hidden_size_1, hidden_size_2, hidden_size_3, dropout_rate):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, hidden_size_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_3, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return nn.functional.log_softmax(x, dim=1)


# All other functions remain the same...
def load_data(batch_size):
    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = MNIST("data", train=True, transform=train_transforms, download=True)
    test_dataset = MNIST("data", train=False, transform=test_transforms, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_model(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_model(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return test_loss, accuracy


def generate_submission_file(model, device, test_loader, filename="submission.csv"):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).cpu().numpy()
            predictions.extend(pred.flatten())

    submission_df = pd.DataFrame(
        {"ID": range(0, len(predictions)), "target": predictions}
    )
    submission_df.to_csv(filename, index=False)
    print(f"Submission file saved as '{filename}'")


def train():
    wandb.init()
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = load_data(config.batch_size)

    model = MLP(
        hidden_size_1=config.hidden_size_1,
        hidden_size_2=config.hidden_size_2,
        hidden_size_3=config.hidden_size_3,
        dropout_rate=config.dropout_rate,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    n_epochs = 30
    best_accuracy = 0
    best_model_path = f"mnist_mlp_model_{wandb.run.id}.pth"

    for epoch in range(1, n_epochs + 1):
        train_loss = train_model(
            model, device, train_loader, optimizer, criterion, epoch
        )
        test_loss, accuracy = evaluate_model(model, device, test_loader, criterion)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_accuracy": accuracy,
            }
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            # Generate submission file for best model
            generate_submission_file(
                model,
                device,
                test_loader,
                f"submission_{wandb.run.id}_{accuracy:.2f}.csv",
            )

    wandb.run.summary["best_accuracy"] = best_accuracy


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="mnist-mlp-sweep")
    wandb.agent(sweep_id, train, count=20)
