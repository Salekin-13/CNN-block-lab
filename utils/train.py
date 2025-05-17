import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, data, num_epochs=10, train_flag=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss() #as the last layer of LeNet is a fully connected layer, crossentropy loss can be used
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    if(train_flag):
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
            
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss:.4f} - Accuracy: {100 * correct / total:.2f}%")

        return model

    else:
        model.eval()
        test_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in data:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _,predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Loss: {test_loss} & Accuracy: {100*correct/total:.2f}%")