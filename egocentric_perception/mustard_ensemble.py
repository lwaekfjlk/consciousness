import torch

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def eval_metric(predictions, labels):
    acc = sum([1 if p == g else 0 for p, g in zip(predictions, labels)]) / len(labels)
    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    return acc, f1, precision, recall

audio_labels = torch.load('./audio_only_labels.pt')
audio_only_predictions = torch.load('./audio_only_predictions.pt')
audio_only_confidences = torch.load('./audio_only_confidences.pt')

text_labels = torch.load('./text_only_labels.pt')
text_only_predictions = torch.load('./text_only_predictions.pt')
text_only_confidences = torch.load('./text_only_confidences.pt')

vision_labels = torch.load('./vision_only_labels.pt')
vision_only_predictions = torch.load('./vision_only_predictions.pt')
vision_only_confidences = torch.load('./vision_only_confidences.pt')


ensembled_predictions = []
labels = []
correct = 0
for idx, (text_pred, vision_pred, audio_pred, text_conf, vision_conf, audio_conf) in enumerate(
    zip(
        text_only_predictions, 
        vision_only_predictions, 
        audio_only_predictions,
        text_only_confidences,
        vision_only_confidences,
        audio_only_confidences,
    )):
    if text_pred is None or vision_pred is None or audio_pred is None:
        continue

    '''
    # oracle prediction
    if text_pred == text_labels[idx] or vision_pred == vision_labels[idx] or audio_pred == audio_labels[idx]:
        correct += 1
        ensembled_predictions.append(text_labels[idx])
        labels.append(text_labels[idx])
        continue
    else:
        ensembled_predictions.append(text_pred)
        labels.append(text_labels[idx])
        continue
    '''

    '''
    # three modality agree with each other
    if text_pred == vision_pred and vision_pred == audio_pred:
        ensembled_predictions.append(text_pred)
        labels.append(text_labels[idx])
        continue
    else:
        pred = [text_pred, vision_pred, audio_pred]
        conf = [text_conf, vision_conf, audio_conf]
        max_index = conf.index(max(conf))
        pred = pred[max_index]
        ensembled_predictions.append(pred)
        labels.append(text_labels[idx])
        continue
    '''

    '''
    pred = [text_pred, vision_pred, audio_pred]
    # check the True number in pred
    true_num = sum([1 if p is True else 0 for p in pred])
    false_num = sum([1 if p is False else 0 for p in pred])
    if true_num > false_num:
        ensembled_predictions.append(True)
    else:
        ensembled_predictions.append(False)
    labels.append(vision_labels[idx])
    '''

    # three modality agree with each other
    if text_pred == vision_pred and vision_pred == audio_pred:
        ensembled_predictions.append(text_pred)
        labels.append(text_labels[idx])
        continue
    else:
        pred = [text_pred, vision_pred, audio_pred]
        conf = [text_conf, vision_conf, audio_conf]
        max_index = conf.index(max(conf))
        pred = pred[max_index]
        ensembled_predictions.append(pred)
        labels.append(text_labels[idx])
        continue


#print(correct / len(text_only_predictions))
acc, f1, precision, recall = eval_metric(ensembled_predictions, labels)
print(acc, f1, precision, recall)