import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.optim as optim


train_transforms = transforms.Compose(
    [
        transforms.RandomAffine(
            degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


test_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


train_dataset = MNIST("data", train=True, transform=train_transforms, download=True)
test_dataset = MNIST("data", train=False, transform=test_transforms, download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return nn.functional.log_softmax(x, dim=1)


model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    amsgrad=False,
)
scheduler = StepLR(optimizer, step_size=10, gamma=0.7)


def smooth_one_hot(targets, n_classes, smoothing=0.1):
    confidence = 1.0 - smoothing
    smooth = smoothing / (n_classes - 1)
    one_hot = torch.zeros(
        (targets.size(0), n_classes), device=targets.device, dtype=torch.float
    )
    one_hot.scatter_(1, targets.unsqueeze(1), confidence)
    one_hot += smooth

    return one_hot


def train(model, device, train_loader, optimizer, epoch, scheduler, smoothing=0.1):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        smoothed_labels = smooth_one_hot(target, n_classes=10, smoothing=smoothing)
        loss = -(smoothed_labels * output).sum(dim=1).mean()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )

    scheduler.step()


def test(model, device, test_loader):
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
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return accuracy


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


n_epochs = 200
best_accuracy = 0

for epoch in range(1, n_epochs + 1):
    train(model, device, train_loader, optimizer, epoch, scheduler)
    accuracy = test(model, device, test_loader)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "mnist_mlp_model.pth")

print(f"Best test accuracy: {best_accuracy:.2f}%")

model.load_state_dict(torch.load("mnist_mlp_model.pth"))
generate_submission_file(model, device, test_loader)
