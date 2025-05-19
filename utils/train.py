import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import time
from datetime import timedelta


def train_model(model, data, classes=None, val=None, num_epochs=10, train_flag=True, opt = 'adam', lr=1e-3, early_stop_pat = 5, reduce_lr_pat = 3, cls_acc=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss() #as the last layer of LeNet is a fully connected layer, crossentropy loss can be used
    
    if(opt == 'sgd'):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=reduce_lr_pat,
        verbose=True,
        min_lr=1e-6
    )

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None


    if train_flag:
        start_time = time.time()

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct, total = 0, 0

            for images, labels in data:
                images, labels = images.to(device), labels.to(device)

                #forward propagation
                outputs = model(images)
                loss = criterion(outputs, labels)

                #backward propagation and optimization
                optimizer.zero_grad() #to avoid gradient accumulation
                loss.backward() #calculate gradients
                optimizer.step() #update model params

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1) #gives the output logits, the max value and the index of that value across the 2nd dimension
                total += labels.size(0) #total number of samples in an epoch
                                        #number of samples in a batch -> labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
            train_loss = running_loss / len(data)

            # Validation step
            model.eval()
            val_loss = 0.0
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for images, labels in val:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= len(val)
            val_acc = 100 * val_correct / val_total

            # Step the scheduler
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current LR: {current_lr:.6f}")


            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"- Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
                  f"- Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_pat:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Training time:", str(timedelta(seconds=round(elapsed_time))))

        # Load best model before returning
        if best_model_state:
            model.load_state_dict(best_model_state)
        return model


    else:
        model.eval()
        test_loss = 0.0
        correct, total = 0, 0

        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        start_time = time.time()

        with torch.no_grad():
            for images, labels in data:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _,predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if cls_acc:
                    if classes is None:
                        raise ValueError("For per class accuracy, please provide class names.")
                    for label, prediction in zip(labels, predicted):
                        class_total[label.item()] += 1
                        if prediction.item() == label.item():
                            class_correct[label.item()] += 1

        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        print("Evaluation time:", str(timedelta(seconds=round(elapsed_time))))

        print(f"Test Loss: {test_loss/len(data)} & Accuracy: {100*correct/total:.2f}%")

        for i, class_name in enumerate(classes):
            C_total = class_total[i]
            correct = class_correct[i]
            acc = 100 * correct / C_total if C_total > 0 else 0
            print(f"{class_name:15}: {acc:.2f}% ({correct}/{C_total})")

        