import torch
import librosa
import csv
import os
from datasets import load_dataset
from tqdm import tqdm
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from scipy.io import wavfile
from datetime import timedelta
import os
import whisper
import librosa
import whisperx


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
    model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")

    bsz = 3
    for i in tqdm(range(0, len(speeches), bsz)):
        # compute attention masks and normalize the waveform if needed
        inputs = feature_extractor(speeches[i: i+bsz], sampling_rate=16000, padding=True, return_tensors="pt")
        logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        labels += [label_dict[model.config.id2label[_id]] for _id in predicted_ids.tolist()]
        print(labels)
    return labels


def preprocess(audio_file):
    device = "cuda" 
    batch_size = 16
    compute_type = "int8"

    model = whisperx.load_model("large-v2", device, compute_type=compute_type, language='en')
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    duration = librosa.get_duration(filename=audio_file)
    if len(result['segments']) != 1:
        import pdb; pdb.set_trace()
    start = result["segments"][0]['start']
    end = result["segments"][-1]['end']
    data, rate = librosa.load(audio_file, res_type='kaiser_fast', duration=1000, sr=22050*2, offset=0)

    start_frame = len(data) * (start / duration)
    end_frame = len(data) * (end / duration)

    filtered_data = data[int(start_frame):int(end_frame)]
    audio_file = audio_file.replace(".aac", ".wav")
    print(audio_file)
    wavfile.write(audio_file, rate,filtered_data)
    return


if __name__ == '__main__':
    dataset = []
    speeches = [] 
    ids = []
    file_path = []
    directory = "./data/audios/utterances_final"
    '''
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".aac"):
                file_path.append(os.path.join(root, file))
                data_id = file.split(".")[0]
                ids.append(data_id)

    for idx, file in enumerate(tqdm(file_path)):
        preprocess(file)
    '''

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                speech = map_to_array(os.path.join(root, file))
                speeches.append(speech)
                data_id = file.split(".")[0]
                ids.append(data_id)    

    emotions = predict(speeches)
    for data_id, emotion in zip(ids, emotions):
        dataset.append([data_id, emotion])
    with open('./data/audios/utterances_final_filtered.csv', 'w') as file:
        writer = csv.writer(file)
        for row in dataset:
            writer.writerow(row)