import random
import numpy as np
import torch
import pandas as pd

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_submission(model, test_loader, device, submission_path="submission.csv", sample_submission_path="/kaggle/input/digit-recognizer/sample_submission.csv"):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    sample_submission = pd.read_csv(sample_submission_path)
    sample_submission['Label'] = predictions
    sample_submission.to_csv(submission_path, index=False)
    print(f"Submission file created: {submission_path}")