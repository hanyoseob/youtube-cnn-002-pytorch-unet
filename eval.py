## 라이브러리를 추가하기
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

import matplotlib.pyplot as plt

## 트레이닝 파라메터를 설정하기
lr = 1e-3
batch_size = 4
num_epoch = 100

num_freq_disp = 10
num_freq_save = 5

data_dir = './dataset/'
ckpt_dir = './checkpoint/'
log_dir = './log/'

result_dir = './result/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 네트워크를 구축하기
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]
            cbr = nn.Sequential(*layers)
            return cbr

        # Encoder part
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)

        # Decoder part
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)

        # self.unpool4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec1_1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)


    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = torch.sigmoid(dec1_1)

        return x

## 데이터 로더를 구축하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, nch=1, transform=None):
        self.data_dir = data_dir
        self.nch = nch
        self.transform = transform

        lst_data = os.listdir(data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_label.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label / 255.0
        input = input / 255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]

        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'label': label, 'input': input}

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.lst_label)

## User-defined transform functions
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}
        return data

class ToNumpy(object):
  def __call__(self, data):
    data = data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    return data

class Normalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        # label = (label - self.mean)/self.std
        input = (input - self.mean)/self.std

        data = {'label': label, 'input': input}
        return data

class Denormalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # label, input = data['label'], data['input']

        # label = (label - self.mean)/self.std
        # input = (input - self.mean)/self.std

        data = (data * self.std) + self.mean

        # data = {'label': label, 'input': input}
        return data

class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, data):
        label, input = data['label'], data['input']

        h, w = input.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
        id_x = np.arange(left, left + new_w, 1).astype(np.int32)

        label = label[id_y, id_x]
        input = input[id_y, id_x]

        data = {'label': label, 'input': input}
        return data

class RandomFlip(object):
  def __call__(self, data):
    label, input = data['label'], data['input']

    if np.random.rand() > 0.5:
      label = np.fliplr(label)
      input = np.flip.lr(input)

    if np.random.rand() > 0.5:
      label = np.flipud(label)
      input = np.flipud(input)

    data  = {'label': label, 'input': input}
    return data

## 네트워크를 저장하거나 불러오는 함수 작성하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               './%s/model_epoch%d.pth' % (ckpt_dir, epoch))

    print('model_epoch%d.pth is saved.' % epoch)

def load(ckpt_dir, net, optim):
    ckpt_lst = os.listdir(ckpt_dir)

    if not ckpt_lst:
        epoch = 0
        return net, optim, epoch

    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    print('%s is loaded.' % ckpt_lst[-1])

    return net, optim, epoch

def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
    if step:
        index.write("<th>step</th>")
    for key, value in fileset.items():
        index.write("<th>%s</th>" % key)
    index.write('</tr>')

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    del fileset['name']

    for key, value in fileset.items():
        index.write("<td><img src='images/%s'></td>" % value)

    index.write("</tr>")
    return index_path

## 학습 시킬 데이터를 불러오기
# transform = transforms.Compose([Normalize(mean=0.5, std=0.5), RandomCrop((256, 256)), ToTensor()])
transform = transforms.Compose([Normalize(mean=0.5, std=0.5), ToTensor()])

dataset_test = Dataset(data_dir=os.path.join(data_dir, 'train'), nch=1, transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)

num_data_test = len(loader_test.dataset)
num_batch_test = round(num_data_test / batch_size)

## 네트워크를 생성하기
net = UNet().to(device)
params = net.parameters()

## 손실함수 등을 설정하기
fn_loss = nn.BCELoss().to(device)

fn_tonumpy = ToNumpy()
fn_denorm = Denormalize(mean=0.5, std=0.5)
fn_class = lambda x: 1.0 * (x > 0.5)

optim = torch.optim.Adam(params, lr=lr)

st_epoch = 0
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

## 트레이닝을 시작하기
# evaluation phase
with torch.no_grad():
    net.eval()
    loss_arr = []

    for batch, data in enumerate(loader_test, 1):
        def should_disp(freq):
            return freq > 0 and (batch % freq == 0 or batch == num_batch_test)

        # forward propagation 하기
        label = data['label'].to(device)
        input = data['input'].to(device)

        output = net(input)

        # backward propagation 하기
        # optim.zero_grad()

        loss = fn_loss(output, label)
        # loss.backward()

        # optim.step()

        # 손실함수를 계산하기
        loss_arr += [loss.item()]

        if should_disp(num_freq_disp):
            print('TEST: BATCH %04d/%04d | LOSS: %.4f' % (batch, num_batch_test, np.mean(loss_arr)))

        label = fn_tonumpy(label)
        input = fn_tonumpy(fn_denorm(input))
        output = fn_tonumpy(fn_class(output))

        for i in range(label.shape[0]):
            name = batch_size * (batch - 1) + i

            fileset = {'name': name,
                      'label': 'label-%04d.png' % name,
                      'input': 'input-%04d.png' % name,
                      'output': 'output-%04d.png' % name}

            label_ = label[i, :, :, :].squeeze()
            input_ = input[i, :, :, :].squeeze()
            output_ = output[i, :, :, :].squeeze()
            result_ = np.concatenate((input_, label_, output_), axis=1)

            plt.imsave(os.path.join(result_dir, fileset['label']), label_, cmap='gray')
            plt.imsave(os.path.join(result_dir, fileset['input']), input_, cmap='gray')
            plt.imsave(os.path.join(result_dir, fileset['output']), output_, cmap='gray')
            plt.imsave(os.path.join(result_dir, 'result-%04d.png' % name), result_, cmap='gray')

            append_index(result_dir, fileset)
