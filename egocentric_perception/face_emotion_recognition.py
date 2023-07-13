import os
import cv2
import csv
import json
from fer import FER
from tqdm import tqdm


def draw_rectangle(image, box_coordinates, color=(0, 255, 0), thickness=2):
    """
    Draws a rectangle on the image using the specified box coordinates.
    
    Args:
        image (numpy.ndarray): The image on which to draw the rectangle.
        box_coordinates (tuple or list): The coordinates of the box in the format (x, y, width, height).
        color (tuple, optional): The color of the rectangle in BGR format. Default is green (0, 255, 0).
        thickness (int, optional): The thickness of the rectangle's lines. Default is 2.
    
    Returns:
        numpy.ndarray: The image with the rectangle drawn.
    """
    for box in box_coordinates:
        x, y, width, height = box 
        cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)
    return image


def predict(frame):
    pred_emotions = []
    pred_boxes = []
    detector = FER()
    detector_results = detector.detect_emotions(frame)
    for detector_result in detector_results:
        boxes = detector_result['box']
        emotions = detector_result['emotions']
        pred_emotion = 'neutral'
        for emotion, score in emotions.items():
            if score > 0.8:
                pred_emotion = emotion
        pred_emotions.append(pred_emotion)
        pred_boxes.append(boxes)

    image_with_rectangle = draw_rectangle(frame, pred_boxes)
    cv2.imwrite('./output_image.jpg', image_with_rectangle)
    print(pred_emotions)
    if 'angry' in pred_emotions:
        import pdb; pdb.set_trace()
    return pred_emotions
    

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