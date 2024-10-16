
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets
from capsulelayers import DenseCapsule, PrimaryCapsule
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
from torchaudio import transforms as T
from torch.utils.data import DataLoader

class CapsuleNet(nn.Module):

    def __init__(self, input_size, classes, routings):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=3, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(1 * 2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(1 * 2)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.maxpool3 = nn.MaxPool2d(1 * 2)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(256, 32, 8, kernel_size=3, stride=1, padding=0)

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=112, in_dim_caps=8,
                                      out_num_caps=classes, out_dim_caps=8, routings=routings)




        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(8*classes, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
            #nn.Linear(1024,4),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)

        x = self.relu(self.conv2(x))

        x = self.maxpool2(x)
        x = self.relu(self.conv3(x))

        x = self.maxpool3(x)

        x = self.primarycaps(x)

        x = self.digitcaps(x)
        length = x.norm(dim=-1)
        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return length, reconstruction.view(-1, *self.input_size)


def caps_loss(y_true, y_pred, x, x_recon, lam_recon):

    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()

    L_recon = nn.MSELoss()(x_recon, x)

    return L_margin + lam_recon * L_recon


def show_reconstruction(model: object, test_loader: object, n_images: object, args: object) -> object:
    import matplotlib.pyplot as plt
    from utils import combine_images
    from PIL import Image
    import numpy as np
    x_recon_li = []

    model.eval()
    for x, y in test_loader:
        x = Variable(x[:min(n_images, x.size(0))].cuda(), volatile=True)
        _, x_recon = model(x)
        x = x.data.cpu()
        x_recon = x_recon.data.cpu()
        data = np.concatenate([x.data, x_recon.data])
        x_recon_li.append(data)
        img = combine_images(np.transpose(data, [0, 2, 3, 1]))
        image = img * 255
        Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
        rrr = Image.fromarray(image.astype(np.uint8))

        print()
        print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
        print('-' * 70)
        #plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png" ),cmap='magma',aspect='auto')
        plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"), cmap='magma', aspect='auto')
        #plt.show()
        break
    #(type(x_recon_li[0][0]))
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 3))

    ax[0].imshow(T.AmplitudeToDB()(torch.from_numpy(x_recon_li[0][0])).squeeze().numpy(), cmap='magma', aspect='auto')
    ax[0].set_title('Reconstructed V')
    ax[0].axis('off')

    ax[1].imshow(T.AmplitudeToDB()(torch.from_numpy(x_recon_li[0][1])).squeeze().numpy(), cmap='magma', aspect='auto')
    ax[1].set_title('Reconstructed O')
    ax[1].axis('off')

    ax[2].imshow(T.AmplitudeToDB()(torch.from_numpy(x_recon_li[0][2])).squeeze().numpy(), cmap='magma', aspect='auto')
    ax[2].set_title('Reconstructed T')
    ax[2].axis('off')

    ax[3].imshow(T.AmplitudeToDB()(torch.from_numpy(x_recon_li[0][3])).squeeze().numpy(), cmap='magma', aspect='auto')
    ax[3].set_title('Reconstructed E')
    ax[3].axis('off')

    plt.tight_layout()
    plt.show()

    #return x_recon_li

    # for x, y in test_loader:
    #     x = Variable(x[:min(n_images, x.size(0))].cuda(), volatile=True)
    #     _, x_recon = model(x)
    #     x = x.data.cpu()
    #     x_recon = x_recon.data.cpu()
    #     for i in range(min(n_images, x.size(0))):
    #         plt.subplot(2, min(n_images, x.size(0)), i + 1)
    #         plt.imshow(x[i].permute(1, 2, 0))
    #         plt.axis('off')
    #         plt.subplot(2, min(n_images, x.size(0)), min(n_images, x.size(0)) + i + 1)
    #         plt.imshow(x_recon[i].permute(1, 2, 0))
    #         plt.axis('off')
    #     plt.show()
    #     break





def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    for x, y in test_loader:
        y = torch.zeros(y.size(0), 4).scatter_(1, y.view(-1, 1), 1.)
        with torch.no_grad():
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x.cuda(), volatile=True), Variable(y.cuda())
        y_pred, x_recon = model(x)
        test_loss += caps_loss(y, y_pred, x, x_recon, args.lam_recon).item() * x.size(0)  # sum up batch loss

        y_pred = y_pred.data.max(1)[1]
        y_true = y.data.max(1)[1]
        correct += y_pred.eq(y_true).cpu().sum()

    test_loss /= len(test_loader.dataset)
    return test_loss, correct / len(test_loader.dataset)


def train(model, train_loader, test_loader, args):

    print('Begin Training' + '-'*70)
    from time import time
    import csv
    logfile = open(args.save_dir + '/log.csv', 'w',newline ='')
    logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss', 'test_loss', 'test_acc'])
    logwriter.writeheader()

    t0 = time()
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    best_val_acc = 0.
    for epoch in range(args.epochs):
        model.train()  # set to training mode
        lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
        ti = time()
        training_loss = 0.0
        for i, (x, y) in enumerate(train_loader):  # batch training
            y = torch.zeros(y.size(0), 4).scatter_(1, y.view(-1, 1), 1.)  # change to one-hot coding
            x, y = Variable(x.cuda()), Variable(y.cuda())  # convert input data to GPU Variable

            optimizer.zero_grad()  # set gradients of optimizer to zero
            y_pred, x_recon = model(x, y)  # forward
            loss = caps_loss(y, y_pred, x, x_recon, args.lam_recon)  # compute loss
            loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
            training_loss += loss.item() * x.size(0)  # record the batch loss
            optimizer.step()  # update the trainable parameters with computed gradients

        # compute validation loss and acc
        val_loss, val_acc = test(model, test_loader, args)
        val_input_acc = val_acc.item()
        logwriter.writerow(dict(epoch=epoch, loss=training_loss / len(train_loader.dataset),
                                test_loss=val_loss, test_acc=val_input_acc))
        #print(val_acc)
        print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, acc=%.4f, time=%ds"
              % (epoch, training_loss / len(train_loader.dataset),
                 val_loss, val_acc, time() - ti))
        if val_acc > best_val_acc:  # update best validation acc and save model
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_dir + '/epoch%d.pkl' % epoch)
            print("best val_acc increased to %.4f" % best_val_acc)
    logfile.close()
    torch.save(model.state_dict(), args.save_dir + '/trained_model.pkl')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
    print("Total time = %ds" % (time() - t0))
    print('End Training' + '-' * 70)
    return model

AUDIO_DIR = "/audio_project/"

class SnoreSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation,
                 target_sample_rate
                 ,num_samples,device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    #len(SSD)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)

        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self,signal, sr): #오직 타켓SR과 원래SR이 다를때 리셈플링)
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self,signal): #여러채널을 가진 오디오채널을 한개채널로 감소
        if signal.shape[0] > 1:    #(2,1000)
            signal = torch.mean(signal, dim = 0, keepdim=True)
        return signal

    def _get_audio_sample_path(self,index):
        fold = f"data"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 1])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 4]


def create_data_loader(train_data, val_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=20)
    val_dataloader = DataLoader(val_data, batch_size=20)
    return train_dataloader, val_dataloader


def load_mpssc(path, download=False, batch_size=100, shift_pixels=2):

    kwargs = {'num_workers': 1, 'pin_memory': True}
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050
    BATCH_SIZE = 20
    #BATCH_SIZE = 30 #for visualizaiton
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        win_length=512,
        hop_length=256,
        n_mels=64,
        norm='slaney'
    )
    transform = torchaudio.transforms.MFCC(
        sample_rate=22050,
        n_mfcc=64,
        melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 64, "center": False},
    )

    ssd_t = SnoreSoundDataset("/Users/asus/Desktop/audio_project/train_labels_aug.csv",
                              AUDIO_DIR,
                              mel_spectrogram,
                              SAMPLE_RATE,
                              NUM_SAMPLES,
                              'cpu')
    ssd_v = SnoreSoundDataset(path,
                              AUDIO_DIR,
                              mel_spectrogram,
                              SAMPLE_RATE,
                              NUM_SAMPLES,
                              'cpu')

    train_dataloader, val_dataloader = create_data_loader(ssd_t, ssd_v, 20)

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    import argparse
    import os

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.0005 * 784, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")  # num_routing should > 0
    parser.add_argument('--shift_pixels', default=2, type=int,
                        help="Number of pixels to shift at most in each direction.")
    parser.add_argument('--data_dir', default='./data',
                        help="Directory of data. If no data, use \'--download\' flag to download it")
    parser.add_argument('--download', action='store_true',
                        help="Download the required data.")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    #print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    train_loader, test_loader = load_mpssc("labeling file", download=False, batch_size=20)
    # test = /test_labels.csv
    # val = /val_labels.csv


    # define model
    model = CapsuleNet(input_size=[1, 64, 87], classes=4, routings=4)
    model.cuda()
    print(model)


    # train or test
    # if args.weights is not None:  # init the model weights with provided one
    #     model.load_state_dict(torch.load(args.weights))
    # if not args.testing:
    #     train(model, train_loader, test_loader, args)
    # else:  # testing
    #     if args.weights is None:
    #         print('No weights are provided. Will test using random initialized weights.')
    #     test_loss, test_acc = test(model=model, test_loader=test_loader, args=args)
    #     print('test acc = %.4f, test loss = %.5f' % (test_acc, test_loss))
    recon = show_reconstruction(model, test_loader, 4, args)
