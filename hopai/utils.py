import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from rich.progress import track

log = logging.getLogger(__name__)


def train(
    model: nn.Module,
    train_dataset,
    test_dataset,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    log_interval: int = 100,
) -> None:
    train_loader = DataLoader(train_dataset, num_workers=10, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, num_workers=10, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    avg_loss = 0.0
    for epoch in range(epochs):
        model.train()
        for i, (x, y) in track(
            enumerate(train_loader), total=len(train_loader), description="Training"
        ):
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                log.info(f"Epoch: {epoch}, Batch: {i}, Loss: {avg_loss / log_interval}")
                avg_loss = 0.0

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                y_hat = model(x)
                _, predicted = torch.max(y_hat.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        log.info(f"Epoch: {epoch}, Accuracy: {correct / total}")
