import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from model import AlexNet
from tqdm import tqdm

def get_train_valid_loader(data_dir, batch_size, download=False):

    normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
    )
    
    transform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        normalize
    ])

    trainset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=transform)

    # validset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)

    # return (train_loader, valid_loader)
    return train_loader


def get_test_loader(data_dir, batch_size, shuffle=True, download=False):

    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
    )

    transform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        normalize
    ])

    dataset = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

if __name__ == "__main__":
    num_classes = 10
    num_epochs = 2
    batch_size = 64
    data_dir = './data'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = get_train_valid_loader(data_dir = data_dir, batch_size = batch_size)

    test_loader = get_test_loader(data_dir = data_dir, batch_size = batch_size)

    model = AlexNet(num_classes).to(device)

    total_step = len(train_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay = 0.005, momentum = 0.9)  

    for epoch in range(num_epochs):

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (images, labels) in progress_bar:  
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    PATH = './alexnet_cifar10.pth'
    torch.save(model.state_dict(), PATH)

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy on the {} test images: {} %'.format(10000, 100 * correct / total))