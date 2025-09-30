import os
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from dataset import RfLegoBeamformerDataset
from model import RfLegoBeamformerModel


def rflego_beamformer_generate_checkpoint_name(args):
    """Generate a descriptive checkpoint name based on training arguments."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        f"{timestamp}_"
        f"batch_{args.batch_size}_"
        f"num_elements_{args.num_elements}_"
        f"dictionary_length_{args.dictionary_length}_"
        f"lr_{args.lr}_"
        f"epochs_{args.epochs}_"
        f"num_layers_{args.num_layers}_"
        f"vis_interval_{args.vis_interval}"
    )


def rflego_beamformer_cosine_loss(pred, target):
    """Cosine similarity loss with sparsity regularization for DoA estimation.
    
    Args:
        pred: Predicted DoA spectrum
        target: Ground truth DoA spectrum
        
    Returns:
        Combined cosine loss and sparsity penalty
    """
    pred_mag = torch.abs(pred)
    target_mag = torch.abs(target)
    
    cos_sim = torch.nn.functional.cosine_similarity(pred_mag, target_mag, dim=-1)
    sparsity_loss = torch.norm(pred_mag, dim=-1).mean()
    
    return 1 - cos_sim.mean() + 0.1 * sparsity_loss


def rflego_beamformer_train(args):
    """Main training function for RF-LEGO Beamformer model."""
    # Set up training environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = RfLegoBeamformerModel(
        dict_length=args.dictionary_length, 
        num_layers=args.num_layers
    ).to(device)

    # Initialize datasets and dataloaders
    train_dataset = RfLegoBeamformerDataset(os.path.join(args.data_dir, "train"))
    val_dataset = RfLegoBeamformerDataset(os.path.join(args.data_dir, "test"))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create checkpoint and log directories
    checkpoint_name = rflego_beamformer_generate_checkpoint_name(args)
    checkpoint_dir = os.path.join(args.save_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    log_path = os.path.join(args.log_dir, checkpoint_name)
    writer = SummaryWriter(log_dir=log_path)

    global_step = 0
    best_val_loss = float("inf")
    logger.info(f"Starting RF-LEGO Beamformer training for {args.epochs} epochs")

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        for batch in train_loader:
            # Prepare inputs
            y = batch["input"].to(device)
            x = batch["label"].to(device)
            A = batch["dictionary"]["dictionary"].to(device)
            
            # Forward pass
            output = model(y, A)
            loss = rflego_beamformer_cosine_loss(output, x)

            # Training step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Log training metrics
            writer.add_scalar("Train_Loss", loss.item(), global_step)
            
            logger.info(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item():.4f}")
            global_step += 1

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_losses = []
            for batch in val_loader:
                y = batch["input"].to(device)
                x = batch["label"].to(device)
                A = batch["dictionary"]["dictionary"].to(device)
                output = model(y, A)
                val_loss = rflego_beamformer_cosine_loss(output, x)
                val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses)
            writer.add_scalar("Validation_Loss", avg_val_loss, global_step)
            logger.info(f"Epoch {epoch}, Validation Loss: {avg_val_loss:.4f}")

            # Save last checkpoint after each epoch
            last_checkpoint_path = os.path.join(checkpoint_dir, "last.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'global_step': global_step
            }, last_checkpoint_path)
            
            # Save best checkpoint if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': avg_val_loss,
                    'global_step': global_step
                }, best_checkpoint_path)
                logger.info(f"Best model saved with validation loss: {avg_val_loss:.4f}")
            
            logger.info(f"Checkpoint saved to {last_checkpoint_path}")
    
    logger.info("RF-LEGO Beamformer training completed successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RF-LEGO Beamformer Model Training")
    
    # Directory arguments
    parser.add_argument("--data_dir", type=str,
                       help="Directory containing train and test data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                       help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="Directory to save training logs")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-2,
                       help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Total number of training epochs")
    
    # Model parameters
    parser.add_argument("--num_elements", type=int, default=8,
                       help="Number of antenna array elements")
    parser.add_argument("--dictionary_length", type=int, default=121,
                       help="Dictionary length (number of DoA angles)")
    parser.add_argument("--num_layers", type=int, default=15,
                       help="Number of unfolded ADMM layers")
    
    args = parser.parse_args()
    
    logger.info("Starting RF-LEGO Beamformer training with arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    rflego_beamformer_train(args)
