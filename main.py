import sys

from tqdm import tqdm
import torch
import pathlib
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from New_VCGmodel import VCG

def seed_set(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
def label_change(label):
    if label == 0:
        return torch.tensor([1, 0], dtype=torch.float32)
    elif label == 1:
        return torch.tensor([0, 1], dtype=torch.float32)
    else:
        raise ValueError("Invalid label")

def covert(values):
    # example : [[0.9, 0.1], [0.3, 0.7]] -> [[1, 0], [0, 1]]
    max_idx = torch.argmax(values, dim=1)
    zero_values = torch.zeros_like(values)
    zero_values[torch.arange(values.size(0)), max_idx] = 1
    return zero_values

seed_set(20230711)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# image root
train_dir = pathlib.Path.cwd().parent / 'new_cats_dogs' / 'train'
test_dir = pathlib.Path.cwd().parent / 'new_cats_dogs' / 'test'
# print(train_dir)

# image convert method
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = ImageFolder(root=train_dir, transform=transform, target_transform=label_change)
test_data_N = ImageFolder(root=test_dir, transform=transform, target_transform=label_change)
# print(train_data.classes)   # ['cat', 'dog']
# print(train_data.class_to_idx)  # {'cat': 0, 'dog': 1}

test_ratio = 0.5
valid_ratio = 0.5

test_data, valid_data = random_split(test_data_N, [test_ratio, valid_ratio])  # get test and valid data

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

epochs = 2
learning_rate = 0.001

model = VCG().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

for epoch in tqdm(range(epochs)):
    model.train()
    for i, (data, label) in enumerate(train_loader):
        '''
        data -> (B, C ,W, H): (32, 3, 224, 224)
        label -> (B, 2)  cat:0 , dog:1
        '''
        data = data.to(device)
        label = label.to(device)

        output = model(data)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 2 == 0:
            print('epoch:[{}/{}] Step:[{}/{}] Loss:{:.4f}'.format(epoch, epochs, i+1, len(train_loader), loss.item()))

    model.eval()
    with torch.no_grad():
        correct_valid = 0
        total_valid = 0
        correct_test = 0
        total_test = 0
        for i, (data, label) in enumerate(valid_loader):
            data, label = data.to(device), label.to(device)
            y_valid = model(data)
            # loss = criterion(y_valid, label)

            y_pred = covert(y_valid.data)

            total_valid += label.shape[0]

            correct_valid += (y_pred == label).sum().item() / 2
        print('total Valid data', total_valid)
        print('Valid Accuracy {}%'.format(100*correct_valid/total_valid))

        for i, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            y_test = model(data)
            # loss = criterion(y_test, label)

            y_pred = covert(y_test.data)
            total_test += label.shape[0]
            correct_test += (y_pred == label).sum().item() / 2
        print('total Test data', total_test)
        print('Test Accuracy {}%'.format(100*correct_test/total_test))
