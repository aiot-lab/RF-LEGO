import os
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from dataset import RfLegoFtDataset
from model import RfLegoFtModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR


def rflego_ft_generate_checkpoint_name(args):
    """Generate a descriptive checkpoint name based on the training arguments, including a timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        f"{timestamp}_"
        f"batch_{args.batch_size}_workers_{args.num_workers}_steps_{args.total_steps}_"
        f"lr_{args.learning_rate}_decay_{args.weight_decay}_step_{args.step_size}_"
        f"gamma_{args.gamma}_sequence_{args.sequence_length}"
    )

def rflego_ft_calculate_loss(clean_fft_pred, clean_fft_gt):
    """Calculate cosine similarity loss between predicted and ground truth frequency spectra."""
    # Separate real and imaginary parts and concatenate
    clean_fft_pred_real = clean_fft_pred.real
    clean_fft_pred_imag = clean_fft_pred.imag
    clean_fft_gt_real = clean_fft_gt.real
    clean_fft_gt_imag = clean_fft_gt.imag
    
    clean_fft_pred_combined = torch.cat((clean_fft_pred_real, clean_fft_pred_imag), dim=-2)
    clean_fft_gt_combined = torch.cat((clean_fft_gt_real, clean_fft_gt_imag), dim=-2)

    # Calculate magnitude and apply cosine similarity loss
    pred_magnitude = torch.linalg.norm(clean_fft_pred_combined, dim=-2)
    gt_magnitude = torch.linalg.norm(clean_fft_gt_combined, dim=-2)
    
    cosine_sim = torch.nn.CosineSimilarity()
    loss = 1 - cosine_sim(pred_magnitude, gt_magnitude).mean()
    
    return loss


def rflego_ft_l0_regularization(model, threshold=1e-5):
    """Approximate L0 regularization by counting weights above a threshold."""
    l0_penalty = 0.0
    for param in model.parameters():
        if param.requires_grad:
            l0_penalty += (param.abs() > threshold).float().sum()
    return l0_penalty

def rflego_ft_train(args):
    """Main training function for RF-LEGO FT model."""
    # Set up training environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_name = rflego_ft_generate_checkpoint_name(args)
    checkpoint_dir = os.path.join(args.save_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, checkpoint_name))

    # Initialize model and datasets
    model = RfLegoFtModel(N=args.sequence_length, device=device)
    train_dataset = RfLegoFtDataset(args.train_data_dir)
    val_dataset = RfLegoFtDataset(args.val_data_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_loss = float("inf")
    current_step = 0
    logger.info(f"Starting RF-LEGO FT training for {args.total_steps} steps")

    # Main training loop
    while current_step < args.total_steps:
        for phase, loader in [('train', train_loader), ('val', val_loader)]:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                accumulated_val_loss = 0.0
                num_val_steps = 0

            for batch in loader:
                # Extract and prepare input data
                noise_signal_real = batch['input'].real.to(device).float()
                noise_signal_imag = batch['input'].imag.to(device).float()
                clean_fft_gt_real = batch['label'].real.to(device).float()
                clean_fft_gt_imag = batch['label'].imag.to(device).float()
                clean_gt_real = batch['label_signal'].real.to(device).float()
                clean_gt_imag = batch['label_signal'].imag.to(device).float()
                noise_fft_real = batch['input_fft'].real.to(device).float()
                noise_fft_imag = batch['input_fft'].imag.to(device).float()
                
                # Ensure correct input shape [batch, 1, sequence_length]
                if noise_signal_real.dim() == 2:
                    noise_signal_real = noise_signal_real.unsqueeze(1)
                    noise_signal_imag = noise_signal_imag.unsqueeze(1)

                if clean_fft_gt_real.dim() == 2:
                    clean_fft_gt_real = clean_fft_gt_real.unsqueeze(1)
                    clean_fft_gt_imag = clean_fft_gt_imag.unsqueeze(1)
                
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass through RF-LEGO FT model
                    clean_fft_pred_real, clean_fft_pred_imag = model(noise_signal_real, noise_signal_imag)
                    
                    # Ensure output shape matches target
                    if clean_fft_gt_real.dim() == 2 and clean_fft_pred_real.dim() == 3:
                        clean_fft_pred_real = clean_fft_pred_real.squeeze(1)
                        clean_fft_pred_imag = clean_fft_pred_imag.squeeze(1)
                    
                    # Convert to complex tensors
                    clean_fft_pred = torch.complex(clean_fft_pred_real, clean_fft_pred_imag)
                    clean_fft_gt = torch.complex(clean_fft_gt_real, clean_fft_gt_imag)
                    clean_gt = torch.complex(clean_gt_real, clean_gt_imag)
                    
                    # Calculate loss
                    primary_loss = rflego_ft_calculate_loss(clean_fft_pred, clean_fft_gt)
                    total_loss = primary_loss

                    if phase == 'train':
                        # Training step
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                        scheduler.step()
                        
                        # Log training metrics
                        current_lr = optimizer.param_groups[0]['lr']
                        writer.add_scalar("Learning_Rate", current_lr, current_step)
                        writer.add_scalar("Train_Loss", primary_loss.item(), current_step)
                        writer.add_scalar("Total_Train_Loss", total_loss.item(), current_step)
                        
                        logger.info(f"Step {current_step}/{args.total_steps}, Loss: {total_loss.item():.6f}")

                        current_step += 1
                        if current_step >= args.total_steps:
                            break
                    else:
                        # Validation step
                        accumulated_val_loss += total_loss.item()
                        num_val_steps += 1

            if phase == 'val' and num_val_steps > 0:
                # Log validation metrics and save best model
                average_val_loss = accumulated_val_loss / num_val_steps
                writer.add_scalar("Validation_Loss", average_val_loss, current_step)
                
                if average_val_loss < best_loss:
                    best_loss = average_val_loss
                    save_path = os.path.join(checkpoint_dir, checkpoint_name + ".pt")
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"Best model saved with validation loss: {best_loss:.6f}")
    
    logger.info("RF-LEGO FT training completed successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RF-LEGO FT Model Training")
    
    # Directory arguments
    parser.add_argument("--save_dir", type=str, default="./checkpoints", 
                       help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs", 
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
    parser.add_argument("--total_steps", type=int, default=100000, 
                       help="Total number of training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-3, 
                       help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                       help="Weight decay for regularization")
    parser.add_argument("--step_size", type=int, default=8000, 
                       help="Step size for learning rate scheduler")
    parser.add_argument("--gamma", type=float, default=0.85, 
                       help="Gamma for learning rate scheduler")
    
    # Model parameters
    parser.add_argument("--sequence_length", type=int, default=256, 
                       help="Input sequence length")
    
    args = parser.parse_args()
    
    logger.info("Starting RF-LEGO FT training with arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    rflego_ft_train(args)