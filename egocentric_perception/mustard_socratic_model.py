import datetime
import json
import os
import re
import time
import csv

import requests
import matplotlib.pyplot as plt
import numpy as np
import openai
import clip
import cv2
from tqdm import tqdm
from PIL import Image
from profanity_filter import ProfanityFilter
from google.cloud import vision
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from fer import FER

gpt_version = "gpt-3.5-turbo-0613"
openai.api_key = os.environ['OPENAI_API_KEY']


def get_text_feats(in_text, batch_size=64):
    start_time = time.time()
    text_tokens = clip.tokenize(in_text).cuda()
    text_id = 0
    text_feats = torch.zeros((len(in_text), clip_feat_dim), dtype=torch.float32).cuda()
    while text_id < len(text_tokens):  # Batched inference.
        batch_size = min(len(in_text) - text_id, batch_size)
        text_batch = text_tokens[text_id:text_id+batch_size]
        with torch.no_grad():
            batch_feats = model.encode_text(text_batch).float()
            batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
            text_feats[text_id:text_id+batch_size, :] = batch_feats
            text_id += batch_size
    end_time = time.time()
    print('get_text_feats: {}'.format(end_time - start_time))
    return text_feats


def get_img_feats(img):
    start_time = time.time()
    img_pil = Image.fromarray(np.uint8(img))
    img_in = preprocess(img_pil)[None, ...]
    with torch.no_grad():
        img_feats = model.encode_image(img_in.cuda()).float()
    img_feats /= img_feats.norm(dim=-1, keepdim=True)
    end_time = time.time()
    print('get_img_feats: {}'.format(end_time - start_time))
    return img_feats


def get_sim(raw_texts, text_feats, img_feats):
    start_time = time.time()
    scores = text_feats @ img_feats.T
    scores = scores.squeeze()
    high_to_low_ids = torch.argsort(scores, descending=True).squeeze()
    high_to_low_texts = [raw_texts[i] for i in high_to_low_ids.tolist()]
    high_to_low_scores = torch.sort(scores, descending=True).values.squeeze()
    end_time = time.time()
    print('nn_text: {}'.format(end_time - start_time))
    return high_to_low_texts, high_to_low_scores.tolist()


def prompt_llm(prompt, max_tokens=64, temperature=0, stop=None):
    response = openai.Completion.create(engine=gpt_version, prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
    return response["choices"][0]["text"].strip()

def prompt_chatllm(prompt, max_tokens=64, temperature=0, stop=None):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=gpt_version,
        messages=messages,
    )
    return response["choices"][0]["message"]["content"].strip()

def load_texts():
    start_time = time.time()
    # Load scene categories from Places365.
    place_categories = np.loadtxt('categories_places365.txt', dtype=str)
    place_texts = []
    for place in place_categories[:, 0]:
        place = place.split('/')[2:]
        if len(place) > 1:
            place = place[1] + ' ' + place[0]
        else:
            place = place[0]
        place = place.replace('_', ' ')
        place_texts.append(place)

    # Load object categories from Tencent ML Images.
    with open('dictionary_and_semantic_hierarchy.txt') as fid:
        object_categories = fid.readlines()
    object_texts = []
    pf = ProfanityFilter()
    # TODO: need to use [1:]
    for object_text in tqdm(object_categories[1:]):
        object_text = object_text.strip()
        object_text = object_text.split('\t')[3]
        safe_list = ''
        for variant in object_text.split(','):
            text = variant.strip()
            if pf.is_clean(text):
                safe_list += f'{text}, '
        safe_list = safe_list[:-2]
        if len(safe_list) > 0:
            object_texts.append(safe_list)
    object_texts = [o for o in list(set(object_texts)) if o not in place_texts]  # Remove redundant categories.
    end_time = time.time()
    print('load_texts: {}'.format(end_time - start_time))
    return place_texts, object_texts


# Zero-shot VLM: classify places.
def get_place(img_feats, place_texts):
    place_topk = 3
    place_feats = get_text_feats([f'Photo of a {p}.' for p in place_texts ])
    sorted_places, places_scores = get_sim(place_texts, place_feats, img_feats)
    return sorted_places[:place_topk]


# Zero-shot VLM: classify objects.
def get_obj(img_feats, object_texts):
    obj_topk = 3
    object_feats = get_text_feats([f'Photo of a {o}.' for o in object_texts])
    sorted_obj_texts, obj_scores = get_sim(object_texts, object_feats, img_feats)
    return sorted_obj_texts[:obj_topk] 


def get_ocr(img):
    text = pytesseract.image_to_string(img)
    return text

def get_ocr(img):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    image = vision.Image(content=img)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return texts

# Zero-shot LM: generate activities.
def get_activity(img_feats, places, objs):
    activity_topk = 3
    activity_texts = []
    prompt = f'''Places: {places[0]}, {places[1]}, or {places[2]}.
    Objects: {objs[0]}, {objs[1]}, {objs[2]}.
    Activities (separate them with /): '''
    gen_res = [prompt_chatllm(prompt, temperature=0.9) for _ in range(activity_topk)]
    for activity in gen_res:
        activities = [a.strip().replace('.', '') for a in activity.split('/')]
        activity_texts += activities

    activity_feats = get_text_feats(activity_texts)
    sorted_activities, activity_scores = get_sim(activity_texts, activity_feats, img_feats)
    return sorted_activities[:activity_topk]


def get_summary(places, objs, activities):
    # Zero-shot LM: generate image summary.
    prompt = f'''I am in a {places[0]}, {places[1]}, or {places[2]}.
I see a {objs[0]}, {objs[1]}, {objs[2]}.
I am {activities[0]}, {activities[1]}, or {activities[2]}.
Question: What am I doing? Answer: I am most likely '''
    summary = prompt_llm(prompt, temperature=0.9)
    return summary


def get_state_world(img):
    verbose = True #@param {type:"boolean"}
    #ocr_text = get_ocr(img)
    #import pdb; pdb.set_trace()
    place_texts, object_texts = load_texts()
    img_feats = get_img_feats(img)
    places = get_place(img_feats, place_texts)
    print(places)
    objs = get_obj(img_feats, object_texts)
    print(objs)
    activities = get_activity(img_feats, places, objs)
    print(activities)
    summary = get_summary(places, objs, activities)
    print(summary)

    state_world = f'''People are in a {places[0]}, {places[1]}, or {places[2]}.
People see a {objs[0]}, {objs[1]}, {objs[2]}.
People are {activities[0]}, {activities[1]}, or {activities[2]}.
People are most likely {summary}. ''' 

    if verbose:
        print(state_world)
    return state_world


def add_face_emotion(img, prompt):
    emotion_prompt = ''
    detector = FER()
    detector_results = detector.detect_emotions(img)
    for detector_result in detector_results:
        emotions = detector_result['emotions']
        for emotion, score in emotions.items():
            if score > 0.7:
                emotion_prompt += 'One person feels {}.\n'.format(emotion)
    if len(emotion_prompt) == 0:
        return prompt
    else:
        if "In the video, people have different emotions:" not in prompt:
            prompt += "In the video, people have different emotions:\n" + emotion_prompt
        else:
            prompt += emotion_prompt
        return prompt


def add_audio(utterance, emotion, prompt):
    prompt += '"{}" is said in a {} tone.\n'.format(utterance, emotion)
    return prompt


def count_files_in_directory(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


def filter_invalid_ans(ans):
    judgement = None
    if 'YES' in ans or 'Yes' in ans:
        judgement = True
    elif 'NO' in ans or 'No' in ans:
        judgement = False
    
    confidence = check_numbers(ans)
    return judgement, confidence


def check_numbers(string):
    pattern = r'[1-5]'
    matches = re.findall(pattern, string)
    
    if matches:
        return int(matches[0])
    else:
        return None


def additional_audio_module(audio_emotion, utterance):
    iter_num = 0
    while True:
        iter_num += 1
        print('audio_module: {}'.format(iter_num))
        if iter_num > 5:
            break
        try:
            prompt = "Think about the relationship between tone emotions like happy / angry / sad / neutral and whether a sentence is sarcastic or not.\n"
            prompt += f"The last utterance '{utterance}' is said in a {audio_emotion} tone.\n"
            prompt += f'Question: Is the last utterance sarcastic? Rate your confidence for your answer from 1-5 and answer with YES or NO: ' 
            print(prompt)
            socratic_res = prompt_chatllm(prompt, temperature=0.9)
            if 'YES' in socratic_res or 'Yes' in socratic_res and check_numbers(socratic_res) is not None:
                break
            elif 'NO' in socratic_res or 'No' in socratic_res and check_numbers(socratic_res) is not None:
                break
        except:
            print('Error, retrying...')
            time.sleep(5)
    return socratic_res


def get_prediction_confidence(prompt):
    ans = None
    prediction = None
    confidence = None

    iter_num = 0
    while True:
        iter_num += 1
        if iter_num > 5:
            print('Trying too many times, just ignore this example.')
            break
        try:
            ans = prompt_chatllm(prompt, temperature=0.9)
            prediction, confidence = filter_invalid_ans(ans)
            if prediction is not None and confidence is not None:
                break
        except:
            print('Error, retrying...')
            time.sleep(5)
    return ans, prediction, confidence


def load_audio_info():
    audio_info = {}
    with open('./data/audios/utterances_final_.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            audio_info[row[0]] = row[-1]
    return audio_info


def load_vision_info():
    vision_info = {}
    with open('./data/frames/face_emotions.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            vision_info[row[0]] = row[1:]
    return vision_info


def add_text(data, prompt):
    utterance = data['utterance']
    context = data['context']
    utterance_speaker = data['speaker']
    context_speaker = data['context_speakers']
    for s, u in zip(context_speaker, context):
        prompt += '{}: {}\n'.format(s, u)
    prompt += '{}: {}\n'.format(utterance_speaker, utterance)
    return prompt


if __name__ == '__main__':
    with open('./sarcasm_data.json', 'r') as f:
        dataset = json.load(f)
    predictions = []
    confidences = []
    gths = []


    audio_info = load_audio_info()
    vision_info = load_vision_info()

    for index, (idx, data) in enumerate(tqdm(dataset.items())):
        prompt = """Here is the final question:\n"""
        prompt = add_text(data, prompt)
        prompt += f'Question: Is the last utterance sarcastic? Rate your confidence for your answer from 1-5 and answer with YES or NO: ' 
        print(prompt)

        ans, prediction, confidence = get_prediction_confidence(prompt)
        gths.append(data['sarcasm'])
        predictions.append(prediction)
        confidences.append(confidence)
        print(ans)

    
    # given gths list and predictions list, compute accuracy
    acc = sum([1 if p == g else 0 for p, g in zip(predictions, gths)]) / len(gths)
    # compute f1 score
    f1 = f1_score(gths, predictions)
    # compute precision and recall
    precision = precision_score(gths, predictions)
    recall = recall_score(gths, predictions)
    print(acc, f1, precision, recall)
    import pdb; pdb.set_trace()