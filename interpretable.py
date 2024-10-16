import numpy as np
import matplotlib.pyplot as plt
import torch
from capsulenet import CapsuleNet
import PIL
from capsulenet import torchaudio
import torchaudio.transforms as T
from torch.autograd import Variable
import torch
import torchaudio
import numpy as np
# from cnn import ResNet
from capsulenet import load_mpssc
import scipy
from torchaudio import transforms
from capsulenet import SnoreSoundDataset,CapsuleNet

import re
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
ANNOTATIONS_FILE = "test_labels.csv"
AUDIO_DIR = "/audio_project/data"


SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 20
mel_spec_transform = T.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        win_length=512,
        hop_length=256,
        n_mels=64,
        norm='slaney'
    )

waveform, sample_rate = torchaudio.load("/Users/asus/Desktop/audio_project/data/test_0001.wav")
mel_spec = mel_spec_transform(waveform).cpu()

mel_spec = mel_spec_transform(waveform)

# Convert Mel spectrogram to decibel scale



model = CapsuleNet(input_size=[1, 64, 87], classes=4, routings=4)
state_dict = torch.load("/Users/asus/Desktop/CapsNet-Pytorch-master/result/train58.5.pkl")
model.load_state_dict(state_dict)
model.to('cuda')
model.eval()

train_loader, test_loader = load_mpssc(ANNOTATIONS_FILE, download=False, batch_size=20)


images, labels = next(iter(test_loader))
images = images.to('cuda')
print(images.size())

# Capsule Activations
with torch.no_grad():
    _, digitcaps_output = model(images)
digitcaps_output_np = digitcaps_output.cpu().numpy()
digitcaps_output_np = np.transpose(digitcaps_output_np, (0, 1, 3, 2))

""" visualize heatmap """
# for i in range(20):
#     image = images[i].cpu().numpy().transpose(1, 2, 0)
#     plt.figure()
#     plt.imshow(digitcaps_output_np[i, 0], cmap='viridis', interpolation='bilinear')
#     plt.title(f'Capsule Activations for Image {i+1}')
#     plt.colorbar()
#     plt.show()
#
#     plt.figure()
#     plt.imshow(image, cmap='magma')
#     plt.title(f'Original Image {i + 1}')
#     plt.show()
# #
fig, axes = plt.subplots(4, 5, figsize=(12, 10))

for i in range(20):
    #image = images[i].cpu().numpy().transpose(1, 2, 0)
    image = T.AmplitudeToDB()(images[i])

    row = i // 5  # 행 인덱스 계산
    col = i % 5  # 열 인덱스 계산

    ax = axes[row, col]
    ax.imshow(image.squeeze().cpu().numpy(), cmap='magma', interpolation='bilinear')
    #ax.imshow(digitcaps_output_np[i, 0], cmap='viridis', interpolation='bilinear')
    #ax.set_title(f'Capsule Activations for Image {i+1}')
    ax.axis('off')

#plt.subplots_adjust(wspace=0.1, hspace=0.3)  # 서브플롯 간 간격 조정


plt.show()


# fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(10, 40))
# for i, (image, caps_activation) in enumerate(zip(visu_images, digitcaps_output_np)):
#     row, col = 0, i
#     # mel_spec_db = T.AmplitudeToDB()(mel_spec)
#     # plt.imshow(mel_spec_db.squeeze().numpy(), cmap='magma')
#     # plt.axis('off')
#     #img = image.cpu().numpy().transpose(1, 2, 0)
#     img = T.AmplitudeToDB()(image)
#     axes[row, col].imshow(img.squeeze().cpu().numpy(), cmap='magma')
#     axes[row, col].set_xticks([])
#     axes[row, col].set_yticks([])
#     axes[row, col].set_title(f'Original Image {i + 1}')
#
#     row, col = 1, i
#     axes[row, col].imshow(caps_activation[0], cmap='viridis', interpolation='nearest')
#     axes[row, col].set_xticks([])
#     axes[row, col].set_yticks([])
#     #axes[row, col].set_title(f'Capsule Activations for Image {i + 1}')
#
# plt.tight_layout()
# plt.show()


# 1 Capsule Activation 시각화
