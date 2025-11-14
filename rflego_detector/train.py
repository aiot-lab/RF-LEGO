import os
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from dataset import RfLegoDetectorDataset
from model import RfLegoDetectorModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.nn import BCEWithLogitsLoss

# Set seed for reproducibility
torch.manual_seed(89)


def rflego_detector_generate_checkpoint_name(args):
    """Generate a descriptive checkpoint name based on training arguments."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        f"{timestamp}_bs{args.batch_size}_lr{args.learning_rate}_wd{args.weight_decay}"
        f"_steps{args.total_steps}_layers{args.num_layers}_hdim{args.hidden_dim}"
    )


def rflego_detector_train(args):
    """Main training function for RF-LEGO Detector model."""
    # Set up training environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_name = rflego_detector_generate_checkpoint_name(args)
    checkpoint_dir = os.path.join(args.save_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, checkpoint_name))

    # Initialize model
    model = RfLegoDetectorModel(
        num_layers=args.num_layers,
        d=args.hidden_dim,
        order=args.order,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        channels=1,
        dropout=args.dropout
    ).to(device)

    # Initialize datasets and dataloaders
    train_dataset = RfLegoDetectorDataset(args.train_data_dir)
    val_dataset = RfLegoDetectorDataset(args.val_data_dir)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers
    )

    # Setup loss, optimizer, and scheduler
    criterion = BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_val_loss = float("inf")
    current_step = 0
    logger.info(f"Starting RF-LEGO Detector training for {args.total_steps} steps")

    # Main training loop
    while current_step < args.total_steps:
        model.train()
        for batch in train_loader:
            # Prepare inputs
            inputs = batch['input'].to(device).float()  # [B, 1, L]
            labels = batch['labels'].to(device).float()  # [B, L]
            
            # Reshape for model: (L, B)
            x = inputs.squeeze(1).permute(1, 0)

            # Forward pass
            logits = model(x)
            loss = criterion(logits.permute(1, 0).squeeze(1), labels)

            # Training step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log training metrics
            writer.add_scalar("Train_Loss", loss.item(), current_step)
            
            logger.info(f"Step {current_step}/{args.total_steps}, Loss: {loss.item():.6f}")

            current_step += 1
            if current_step >= args.total_steps:
                break

        # Validation after each epoch
        model.eval()
        val_loss, count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device).float()
                labels = batch['labels'].to(device).float()
                x = inputs.squeeze(1).permute(1, 0)
                
                # Forward pass
                logits = model(x)
                loss = criterion(logits.permute(1, 0).squeeze(1), labels)
                
                val_loss += loss.item()
                count += 1

        # Log validation metrics
        avg_val_loss = val_loss / count
        writer.add_scalar("Validation_Loss", avg_val_loss, current_step)
        logger.info(f"Validation Loss: {avg_val_loss:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(checkpoint_dir, checkpoint_name + ".pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Best model saved with validation loss: {best_val_loss:.6f}")
    
    logger.info("RF-LEGO Detector training completed successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RF-LEGO Detector Model Training")
    
    # Directory arguments
    parser.add_argument("--save_dir", type=str,
                       help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str,
                       help="Directory to save training logs")
    parser.add_argument("--train_data_dir", type=str,
                       help="Directory containing training data")
    parser.add_argument("--val_data_dir", type=str,
                       help="Directory containing validation data")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--total_steps", type=int, default=10000,
                       help="Total number of training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for regularization")
    parser.add_argument("--step_size", type=int, default=1000,
                       help="Step size for learning rate scheduler")
    parser.add_argument("--gamma", type=float, default=0.9,
                       help="Gamma for learning rate scheduler")
    
    # Model parameters
    parser.add_argument("--num_layers", type=int, default=1,
                       help="Number of state space layers")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Hidden dimension of the model")
    parser.add_argument("--order", type=int, default=256,
                       help="State space order (dimension of state)")
    parser.add_argument("--dt_min", type=float, default=8e-5,
                       help="Minimum discretization step size")
    parser.add_argument("--dt_max", type=float, default=1e-1,
                       help="Maximum discretization step size")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout probability")
    
    args = parser.parse_args()
    
    logger.info("Starting RF-LEGO Detector training with arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    rflego_detector_train(args)
