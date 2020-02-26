# src/utils/utils.py

from pathlib import Path

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def early_stop_callback() -> EarlyStopping:
    return EarlyStopping(patience=10, verbose=True, monitor="mean_val_loss")


def test_checkpoint_callback(filepath: Path) -> ModelCheckpoint:
    return ModelCheckpoint(filepath=filepath, save_best_only=True, monitor="mean_val_loss")


def train_checkpoint_callback(filepath: Path) -> ModelCheckpoint:
    return ModelCheckpoint(filepath=filepath, save_best_only=True, monitor="mean_val_loss")
