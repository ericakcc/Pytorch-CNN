import time
import torch

def train_one_epoch(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / len(train_loader), 100.0 * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(val_loader), 100.0 * correct / total

def train_and_validate(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs=15):
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    best_val_acc = 0.0
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()
        print(f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        # save best weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
    total_time = time.time() - start_time
    print(f"Training Completed in {total_time:.2f} seconds")
    return model