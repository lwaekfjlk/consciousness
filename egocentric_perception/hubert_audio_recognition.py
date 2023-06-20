import torch
import librosa
import csv
import os
from datasets import load_dataset
from tqdm import tqdm
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

label_dict = {
    'neu': 'neutral',
    'hap': 'happy',
    'ang': 'angry',
    'sad': 'sad',
}

def map_to_array(input_file):
    speech, _ = librosa.load(input_file, sr=16000, mono=True)
    return speech


def predict(speeches):
    labels = []
    model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-er")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-er")

    bsz = 3
    for i in tqdm(range(0, len(speeches), bsz)):
        # compute attention masks and normalize the waveform if needed
        inputs = feature_extractor(speeches[i: i+bsz], sampling_rate=16000, padding=True, return_tensors="pt")
        logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        labels += [label_dict[model.config.id2label[_id]] for _id in predicted_ids.tolist()]
        print(labels)
    return labels

if __name__ == '__main__':
    dataset = []
    speeches = [] 
    ids = []
    directory = "./data/audios/utterances_final"
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".aac"):
                speech = map_to_array(os.path.join(root, file))
                speeches.append(speech)
                data_id = file.split(".")[0]
                ids.append(data_id)

    emotions = predict(speeches)
    for data_id, emotion in zip(ids, emotions):
        dataset.append([data_id, emotion])
    with open('./data/audios/utterances_final_.csv', 'w') as file:
        writer = csv.writer(file)
        for row in dataset:
            writer.writerow(row)