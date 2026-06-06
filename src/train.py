"""
train.py

Train the GameTune Transformer on tokenized NES-MDB sequences.
"""

from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from src.dataset import GameTuneDataset
from src.model import GameTuneModel
from src.model import GameTuneModelConfig


# PAD is vocabulary entry 0.
# We ignore it when computing training loss.
PAD_TOKEN_ID = 0


def load_config(path: Path) -> dict:
    """
    Load YAML configuration.
    """
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_model(config: dict) -> GameTuneModel:
    """
    Build a model from configuration values.
    """
    model_config = GameTuneModelConfig(**config["model"])
    return GameTuneModel(model_config)


def estimate_loss(
    model: GameTuneModel,
    dataloader: DataLoader,
    device: str,
    eval_batches: int,
) -> float:
    """
    Evaluate the model on a small number of batches.

    We only evaluate a few batches because full validation
    would slow training significantly.
    """
    model.eval()

    losses: list[float] = []

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(dataloader):
            if batch_index >= eval_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model(inputs)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=PAD_TOKEN_ID,
            )

            losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


def save_checkpoint(
    path: Path,
    model: GameTuneModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
) -> None:
    """
    Save everything needed to resume training later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def train(config: dict) -> None:
    """
    Main training loop.
    """
    device = config["training"]["device"]

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    train_dataset = GameTuneDataset(
        Path(config["data"]["train_root"]),
        context_size=config["model"]["context_size"],
    )

    test_dataset = GameTuneDataset(
        Path(config["data"]["test_root"]),
        context_size=config["model"]["context_size"],
    )

    # ---------------------------------------------------------
    # Data loaders
    # ---------------------------------------------------------
    #
    # DataLoader handles:
    #   - batching
    #   - shuffling
    #   - multiprocessing
    #

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = create_model(config).to(device)

    # AdamW is the standard optimizer for GPT-style models.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    checkpoint_path = (
        Path(config["checkpoints"]["output_dir"])
        / config["checkpoints"]["checkpoint_name"]
    )

    print(f"train_samples: {len(train_dataset)}")
    print(f"test_samples: {len(test_dataset)}")
    print(f"parameters: {model.parameter_count():,}")
    print(f"device: {device}")

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------

    step = 0
    model.train()

    while step < config["training"]["max_steps"]:
        for inputs, targets in train_loader:
            if step >= config["training"]["max_steps"]:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            # -------------------------------------------------
            # Forward pass
            # -------------------------------------------------
            #
            # Model predicts the next token at every position.
            #

            logits, _ = model(inputs)

            # -------------------------------------------------
            # Loss
            # -------------------------------------------------
            #
            # Compare predicted tokens against expected tokens.
            #

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=PAD_TOKEN_ID,
            )

            # -------------------------------------------------
            # Backpropagation
            # -------------------------------------------------
            #
            # Compute gradients and update parameters.
            #

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1

            # -------------------------------------------------
            # Logging
            # -------------------------------------------------

            if step % 10 == 0:
                print(
                    f"step {step}: "
                    f"train_loss={loss.item():.4f}"
                )

            # -------------------------------------------------
            # Validation
            # -------------------------------------------------

            if step % config["training"]["eval_interval"] == 0:
                eval_loss = estimate_loss(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                    eval_batches=config["training"]["eval_batches"],
                )

                print(
                    f"step {step}: "
                    f"eval_loss={eval_loss:.4f}"
                )

            # -------------------------------------------------
            # Checkpoints
            # -------------------------------------------------

            if step % config["training"]["checkpoint_interval"] == 0:
                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=config,
                )

                print(
                    f"saved checkpoint: "
                    f"{checkpoint_path}"
                )

    # Final checkpoint after training completes.
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=step,
        config=config,
    )

    print()
    print(f"finished training at step {step}")
    print(f"saved checkpoint: {checkpoint_path}")


def main() -> None:
    """
    Entry point.
    """
    parser = argparse.ArgumentParser(
        description="Train the GameTune Transformer."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yml"),
        help="Path to config file.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()

