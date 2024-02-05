import torch


def checkpoint(model, filename):
    """Save model state dict to file."""
    torch.save(model.state_dict(), filename)


def resume(model, filename):
    """Load model state dict from file."""
    model.load_state_dict(torch.load(filename))


def print_progress(epoch, batches_trained, total_batches, bar_length=20):
    """Print progress bar."""
    progress = batches_trained / total_batches
    percent = int(progress * 100)
    num_blocks = int(bar_length * progress)
    bar = "=" * num_blocks + "-" * (bar_length - num_blocks)
    progress_str = f"Epoch {epoch} - Batches: {batches_trained}/{total_batches} - {percent}% [{bar}]"
    print(progress_str, end="\r")
