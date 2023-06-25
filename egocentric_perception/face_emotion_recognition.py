import os
import cv2
import csv
import json
from fer import FER
from tqdm import tqdm

def predict(frame):
    predictions = []
    detector = FER()
    detector_results = detector.detect_emotions(frame)
    for detector_result in detector_results:
        emotions = detector_result['emotions']
        for emotion, score in emotions.items():
            if score > 0.5:
                predictions.append(emotion)
    # delete duplicates
    predictions = list(set(predictions))
    return predictions
    

if __name__ == '__main__':
    with open('./sarcasm_data.json', 'r') as f:
        sarcasm_data = json.load(f)

    dataset = []
    predictions = []
    directory = "./data/frames/utterances_final/"
    for idx, data in tqdm(sarcasm_data.items()):
        dir = directory + idx
        frame_num = len(os.listdir(dir))
        video_prediction = []
        for frame_id in range(1, frame_num+1, 20):
            frame_name = str(frame_id).zfill(5) + '.jpg'
            frame = cv2.imread(os.path.join(dir, frame_name))
            frame_prediction = predict(frame)
            video_prediction += frame_prediction
        predictions.append(video_prediction)
        print(video_prediction)  

    for idx, prediction in zip(sarcasm_data.keys(), predictions):
        dataset.append([idx] + [e for e in prediction])

    with open('./data/frames/face_emotions.csv', 'w') as file:
        writer = csv.writer(file)
        for row in dataset:
            writer.writerow(row)