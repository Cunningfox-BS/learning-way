import torch
import os
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms


BATCH_SIZE=256
EPOCH=100


class AlexNet(nn.Module):
    def __init__(self,num_classes=2):
        super(AlexNet, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,48,kernel_size=11),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(48,128,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(128,192,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.classifier=nn.Sequential(
            nn.Linear(6*6*128,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048,num_classes)
        )

    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,start_dim=1)
        x=self.classifier(x)

        return x

normalize=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
path_1=r"C:\train"
trans_1=transforms.Compose([
    transforms.Resize((65,65)),
    transforms.ToTensor(),
    normalize
])
train_set=ImageFolder(root=path_1,transform=trans_1)
train_loader=torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)

path_2 = r'C:\test'
trans_2 = transforms.Compose([
    transforms.Resize((65, 65)),
    transforms.ToTensor(),
    normalize,
])
test_data = ImageFolder(root=path_2, transform=trans_2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=True,num_workers=0)


path_3=r'C:\valid'
valid_data=ImageFolder(root=path_2,transform=trans_2)
valid_loader=torch.utils.data.DataLoader(valid_data,batch_size=BATCH_SIZE,
                                         shuffle=True,num_workers=0)



model=AlexNet()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0005)


def train_model(model,train_loader,optimizer,epoch):
    train_loss=0
    model.train()
    for batch_index,(data,label) in enumerate(train_loader):
        optimizer.zero_grad()
        output=model(data)
        loss=F.cross_entropy(output,label)
        loss.backward()
        optimizer.step()
        if batch_index%300==0:
            train_loss=loss.item()
            print("Train Epoch:{}\ttrain loss:{:.6f}".format(epoch,loss.item()))
    return train_loss


def test_model(model,test_loader):
    model.eval()
    correct=0
    test_loss=0

    with torch.no_grad():
        for data,label in test_loader:
            output=model(data)
            test_loss+=F.cross_entropy(output,label).item()
            pred=output.argmax(dim=1)
            correct+=pred.eq(label.view_as(pred)).sum().item()
        test_loss/=len(test_loader.dataset)
        print(test_loss)
        print('Test_average_loss:{:.4f},Accuracy:{:3f}\n'.format(
            test_loss, 100 * correct / len(test_loader.dataset)
        ))
        acc = 100 * correct / len(test_loader.dataset)

        return test_loss,acc


list = []
Train_Loss_list = []
Valid_Loss_list = []
Valid_Accuracy_list = []


for epoch in range(1, EPOCH + 1):
    train_loss = train_model(model,train_loader,optimizer,epoch)
    Train_Loss_list.append(train_loss)
    torch.save(model, r'C:\save_model\model%s.pth' % epoch)


    test_loss,acc = test_model(model, valid_loader)
    Valid_Loss_list.append(test_loss)
    Valid_Accuracy_list.append(acc)
    list.append(test_loss)


min_num = min(list)
min_index = list.index(min_num)

print('model%s' % (min_index + 1))
print('验证集最高准确率： ')
print('{}'.format(Valid_Accuracy_list[min_index]))


model = torch.load(r'C:\save_model\model%s.pth' % (min_index + 1))
model.eval()

accuracy = test_model(model, test_loader)
print('测试集准确率')
print('{}%'.format(accuracy))


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


x1 = range(0, EPOCH)
y1 = Train_Loss_list
y2 = Valid_Loss_list
y3 = Valid_Accuracy_list


plt.subplot(221)

plt.plot(x1, y1, '-o')

plt.ylabel('训练集损失')
plt.xlabel('轮数')

plt.subplot(222)
plt.plot(x1, y2, '-o')
plt.ylabel('验证集损失')
plt.xlabel('轮数')

plt.subplot(212)
plt.plot(x1, y3, '-o')
plt.ylabel('验证集准确率')
plt.xlabel('轮数')


plt.show()
































