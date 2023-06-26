import torch

audio_labels = torch.load('./audio_only_labels.pt')
audio_only_predictions = torch.load('./audio_only_predictions.pt')
audio_only_confidences = torch.load('./audio_only_confidences.pt')

text_labels = torch.load('./text_only_labels.pt')
text_only_predictions = torch.load('./text_only_predictions.pt')
text_only_confidences = torch.load('./text_only_confidences.pt')

vision_labels = torch.load('./vision_only_labels.pt')
vision_only_predictions = torch.load('./vision_only_predictions.pt')
vision_only_confidences = torch.load('./vision_only_confidences.pt')

true_labels = 0
false_labels = 0
for label in text_labels:
    if label is True:
        true_labels += 1
    else:
        false_labels += 1
print(true_labels, false_labels)

prediction_bins = {1: [], 2: [], 3: [], 4: [], 5: []}
label_bins = {1: [], 2: [], 3: [], 4: [], 5: []}

for label, prediction, confidence in zip(vision_labels, vision_only_predictions, vision_only_confidences):
    if confidence is not None:
        prediction_bins[confidence].append(prediction)
        label_bins[confidence].append(label)

acc_bins = {}
for i in range(1, 6):
    # calculate accuracy for each bin
    assert len(prediction_bins[i]) == len(label_bins[i])
    acc_bins[i] = sum([1 if p == g else 0 for p, g in zip(prediction_bins[i], label_bins[i])]) / (len(label_bins[i]) + 1e-9)
    
print(acc_bins)
import pdb; pdb.set_trace()