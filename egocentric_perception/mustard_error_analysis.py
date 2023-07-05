import torch
import json
import csv
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from mustard_socratic_model import load_audio_info, load_vision_info


def eval_metric(predictions, labels):
    acc = sum([1 if p == g else 0 for p, g in zip(predictions, labels)]) / len(labels)
    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    return acc, f1, precision, recall


def load_audio_info():
    audio_info = {}
    with open('./data/audios/utterances_final_.csv', 'r') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            audio_info[idx] = row[-1]
    return audio_info


def load_vision_info():
    vision_info = {}
    with open('./data/frames/face_emotions.csv', 'r') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            vision_info[idx] = row[1:]
    return vision_info


with open('./sarcasm_data.json', 'r') as f:
    data = json.load(f)

sarcasm_data = []
for key, value in data.items():
    sarcasm_data.append(value)

audio_labels = torch.load('./audio_only_labels.pt')
audio_only_predictions = torch.load('./audio_only_predictions.pt')
audio_only_confidences = torch.load('./audio_only_confidences.pt')

text_labels = torch.load('./text_only_labels.pt')
text_only_predictions = torch.load('./text_only_predictions.pt')
text_only_confidences = torch.load('./text_only_confidences.pt')

vision_labels = torch.load('./vision_only_labels.pt')
vision_only_predictions = torch.load('./vision_only_predictions.pt')
vision_only_confidences = torch.load('./vision_only_confidences.pt')

all_modality_labels = torch.load('./all_modality_labels.pt')
all_modality_predictions = torch.load('./all_modality_predictions.pt')
all_modality_confidences = torch.load('./all_modality_confidences.pt')



ensembled_predictions = []
labels = []
correct = 0
count = 0
count_true = 0
count_false = 0
indexes = []
for idx, (text_pred, vision_pred, audio_pred, all_modality_pred, text_conf, vision_conf, audio_conf, all_modality_conf) in enumerate(
    zip(
        text_only_predictions, 
        vision_only_predictions, 
        audio_only_predictions,
        all_modality_predictions,
        text_only_confidences,
        vision_only_confidences,
        audio_only_confidences,
        all_modality_confidences,
    )):
    if text_pred is None or vision_pred is None or audio_pred is None or all_modality_predictions is None:
        continue

    true_label = all_modality_labels[idx]
    if true_label == text_pred and true_label != vision_pred and true_label != audio_pred and true_label == all_modality_pred:
        count += 1
        if true_label is True:
            count_true += 1
        else:
            count_false += 1
        indexes.append(idx)
        print(sarcasm_data[idx])

print(count)
print(count_true)
print(count_false)

vision_info = load_vision_info()
audio_info = load_audio_info()

from collections import Counter

vision_pred_dict = {}
for idx in range(690):
    vision_emotions = vision_info[idx]
    for vision_emotion in vision_emotions:
        if vision_emotion not in vision_pred_dict.keys():
            vision_pred_dict[vision_emotion] = {True: 0, False: 0}
        vision_pred = vision_only_predictions[idx]
        if vision_pred is not None:
            vision_pred_dict[vision_emotion][vision_pred] += 1

audio_pred_dict = {}
for idx in range(690):
    audio_emotion = audio_info[idx]
    if audio_emotion not in audio_pred_dict.keys():
        audio_pred_dict[audio_emotion] = {True: 0, False: 0}
    audio_pred = audio_only_predictions[idx]
    if audio_pred is not None:
        audio_pred_dict[audio_emotion][audio_pred] += 1

text_pred_dict = Counter(text_only_predictions)
text_label_dict = Counter(text_labels)


import pdb; pdb.set_trace()

audio_res = []
vision_res = []
for idx in indexes:
    audio_res.append(audio_info[idx])
    vision_res += vision_info[idx]
    print(audio_info[idx])
    print(vision_info[idx])

from collections import Counter
counter = Counter(res)
print(counter)