import torch
import torchaudio
import numpy as np
# from cnn import ResNet

from torchaudio import transforms
from capsulenet import SnoreSoundDataset,CapsuleNet

import re
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
ANNOTATIONS_FILE = "/Users/asus/Desktop/audio_project/test_labels.csv"
AUDIO_DIR = "/Users/asus/Desktop/audio_project"
class_mapping = [
    "V",
    "O",
    "T",
    "E",
]


# def predict(model, input, target, class_mapping):
#     model.eval()
#     with torch.no_grad():
#         predictions = model(input)
#         # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
#         predicted_index = predictions[0].argmax(0)
#
#         predicted = class_mapping[predicted_index]
#         expected = class_mapping[target]
#     return predicted_index,target,predicted, expected, predictions


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        #batch_size, channel, freq = predictions.size()
        #predictions = predictions.reshape(batch_size, 192 * 4)
        #print(predictions.size())
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        #print(predictions)
        predicted_index = predictions[0].argmax(1)
        #print(predicted_index)

        #predicted = class_mapping[predicted_index]
        #expected = class_mapping[target]
    #return predicted_index,target,predicted, expected, predictions
    return predicted_index, target, predictions


if __name__ == "__main__":
    # load back the model

    device = torch.device("cuda")
    model = CapsuleNet(input_size=[1, 64, 87 ], classes=4, routings=4)
    state_dict = torch.load("/Users/asus/Desktop/CapsNet-Pytorch-master/result/train58.5.pkl")
    model.load_state_dict(state_dict)
    model.cuda()

    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050
    BATCH_SIZE = 15
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

    ssd = SnoreSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            NUM_SAMPLES,
                            NUM_SAMPLES,
                            'cpu')

y_true = []
y_pred = []

with open("result.csv", 'w') as csv_file:
    for i in range(263):
        input, target = ssd[i][0].cuda(), ssd[i][1]  # [batch size, num_channels, fr, time]
        input.unsqueeze_(0)
        # make an inference
        predicted, expected, predictions = predict(model, input, target,
                                                   class_mapping)
        y_true.append(expected)
        y_pred.append(predicted)


print(f"y_pred={y_pred.__len__()}")
print(f"y_true={y_true.__len__()}")
y_pred_n = []
for i in y_pred:
    y_pred_n.append(i.tolist())


cm = confusion_matrix(y_true, y_pred_n, labels=[0, 1, 2, 3])
print(cm)
TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP
num_classes = 4
TN = []
for i in range(num_classes):
    temp = np.delete(cm, i, 0)
    temp = np.delete(temp, i, 1)
    TN.append(sum(sum(temp)))
l = 262

precision = TP / (TP + FP)
recall = TP / (TP + FN)
# URA = recall.sum / 4


print(precision)
print(recall)
URA = (recall[0] + recall[1] + recall[2] + recall[3]) / 4
print(f"UAR:{URA}%")
classes = ['V','O','T','E']

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()