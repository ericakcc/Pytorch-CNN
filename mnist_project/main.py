import argparse
import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Import custom modules
from data.dataset import MyDataset, get_transforms
from training.trainer import train_and_validate
from utils.common import set_seed, create_submission

# Model factory: select the model based on argument
def get_model(model_name):
    if model_name == 'ModuleListCNN':
        from models.modulelistcnn import ModuleListCNN
        return ModuleListCNN()
    elif model_name == 'SimpleCNN':
        from models.simplecnn import SimpleCNN
        return SimpleCNN()
    else:
        raise ValueError(f"Model {model_name} not supported.")

def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Use GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load CSV data (modify paths as needed)
    train_csv_path = args.train_csv if args.train_csv else "/kaggle/input/digit-recognizer/train.csv"
    test_csv_path = args.test_csv if args.test_csv else "/kaggle/input/digit-recognizer/test.csv"
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    print("Train CSV shape:", train_df.shape)
    print("Test CSV shape:", test_df.shape)

    # Create data transforms for training and testing
    train_transform = get_transforms(is_train=True, rotation=args.rotation)
    test_transform = get_transforms(is_train=False)

    # Create Datasets and DataLoaders
    full_train_dataset = MyDataset(train_df, transform=train_transform, is_test=False)
    test_dataset = MyDataset(test_df, transform=test_transform, is_test=True)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    # Use test_transform for validation set
    val_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model
    model = get_model(args.model)
    model = model.to(device)

    # Define Loss function and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    # Train and validate the model
    trained_model = train_and_validate(model, train_loader, val_loader, optimizer, criterion, scheduler,
                                       device, num_epochs=args.epochs)

    # If specified, create the submission file
    if args.submit:
        create_submission(trained_model, test_loader, device, submission_path=args.submission_path, 
                          sample_submission_path=args.sample_submission)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MNIST Multi-Model Training and Testing Project")
    parser.add_argument('--model', type=str, default="ModuleListCNN", help="Model name: ModuleListCNN or SimpleCNN")
    parser.add_argument('--seed', type=int, default=2023, help="Random seed")
    parser.add_argument('--rotation', type=int, default=15, help="Random rotation angle for training data")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--scheduler_step', type=int, default=5, help="Step size for learning rate scheduler")
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help="Decay factor for learning rate scheduler")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--submit', action='store_true', help="Generate submission file")
    parser.add_argument('--submission_path', type=str, default="submission.csv", help="Path to save the submission file")
    parser.add_argument('--sample_submission', type=str, default="/kaggle/input/digit-recognizer/sample_submission.csv", help="Path to sample_submission.csv")
    parser.add_argument('--train_csv', type=str, default="", help="Path to training CSV file")
    parser.add_argument('--test_csv', type=str, default="", help="Path to test CSV file")
    args = parser.parse_args()

    main(args)