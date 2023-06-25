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
import logging
import cv2
from tqdm import tqdm
from PIL import Image
from profanity_filter import ProfanityFilter
from google.cloud import vision
import torch
import asyncio
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from fer import FER
from openai_prompting import generate_from_openai_chat_completion
from utils import lm_config

# Create a console handler
console = logging.StreamHandler()
console.setLevel(logging.WARNING)

# Add the handler to the root logger
logging.getLogger().addHandler(console)

# Now this should output a warning
logging.warning('This is a warning')

gpt_version = "gpt-3.5-turbo-0613"
openai.api_key = os.environ['OPENAI_API_KEY']


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


def build_text_prompt(dataset):
    prompts = []
    for index, (idx, data) in enumerate(tqdm(dataset.items())):
        prompt = "Here is the final question:\n"
        utterance = data['utterance']
        context = data['context']
        utterance_speaker = data['speaker']
        context_speaker = data['context_speakers']
        for s, u in zip(context_speaker, context):
            prompt += '{}: {}\n'.format(s, u)
        prompt += '{}: {}\n'.format(utterance_speaker, utterance)
        prompt += f'Question: Is the last utterance sarcastic? Rate your confidence for your answer from 1-5 and answer with YES or NO: ' 
        prompts.append(prompt)
    return prompts


def build_audio_prompt(dataset, audio_info):
    prompts = []
    for index, (idx, data) in enumerate(tqdm(dataset.items())):
        utterance = data['utterance']
        audio_emotion = audio_info[idx]
        # similar prompt with additional_audio_module
        prompt = "Think about the relationship between tone emotions like happy / angry / sad / neutral and whether a sentence is sarcastic or not.\n"
        prompt += f"The last utterance '{utterance}' is said in a {audio_emotion} tone.\n"
        prompt += f'Question: Is the last utterance sarcastic? Rate your confidence for your answer from 1-5 and answer with YES or NO: ' 
        prompts.append(prompt)
    return prompts


def build_vision_prompt(dataset, vision_info):
    prompts = []
    for index, (idx, data) in enumerate(tqdm(dataset.items())):
        utterance = data['utterance']
        vision_emotion = ' ,'.join(vision_info[idx])
        prompt = "Think about the relationship between facial expressions like happy / angry / sad / neutral and whether a sentence is sarcastic or not.\n"
        prompt += f"The last utterance '{utterance}' is said with a {vision_emotion} face.\n"
        prompt += f'Question: Is the last utterance sarcastic? Rate your confidence for your answer from 1-5 and answer with YES or NO: ' 
        prompts.append(prompt)
    return prompts


def generate(prompts):
    # config for generation
    model_name = 'gpt-3.5-turbo'
    max_output_len = 512
    temp = 0.2
    p = 0.8
    config = lm_config.LMConfig(provider='openai', model=model_name)

    # Initialize lists with None
    predictions = [None for _ in range(len(prompts))]
    confidences = [None for _ in range(len(prompts))]

    # Run the loop until all predictions are filled
    iter_num = 0
    while iter_num <= 5:
        iter_num += 1
        gen_ans = []
        for i in range(0, len(prompts), 5):
            while True:
                try:
                    gen_ans += asyncio.run(generate_from_openai_chat_completion(
                        prompts[i:i+5],
                        config,
                        temperature=temp,
                        max_tokens=max_output_len,
                        top_p=p,
                        requests_per_minute=100,
                    ))
                    break
                except:
                    print('Failed! Sleep 5 seconds and try again.')
                    time.sleep(5)
        new_prompts = []
        new_indices = []
        for idx, ans in enumerate(gen_ans):
            prediction, confidence = filter_invalid_ans(ans)
            if prediction is None or confidence is None:
                new_prompts.append(prompts[idx])
                new_indices.append(idx)
            else:
                predictions[idx] = prediction  # update the prediction at this index
                confidences[idx] = confidence  # update the confidence at this index
        if not new_prompts:  # break the loop if we've successfully processed all prompts
            break
        else:
            prompts = new_prompts  # replace old prompts with new ones that got None as prediction
    return predictions, confidences


def eval_metric(predictions, labels):
    acc = sum([1 if p == g else 0 for p, g in zip(predictions, labels)]) / len(labels)
    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    return acc, f1, precision, recall


if __name__ == '__main__':
    with open('./sarcasm_data.json', 'r') as f:
        dataset = json.load(f)
        labels = [data['sarcasm'] for idx, data in dataset.items()]

    predictions = []
    confidences = []

    audio_info = load_audio_info()
    vision_info = load_vision_info()

    text_prompts = build_text_prompt(dataset)
    audio_prompts = build_audio_prompt(dataset, audio_info)
    vision_prompts = build_vision_prompt(dataset, vision_info)

    predictions, confidences = generate(text_prompts)

    import pdb; pdb.set_trace()
    predictions = [p for p, c, l in zip(predictions, confidences, labels) if p is not None]
    confidences = [c for p, c, l in zip(predictions, confidences, labels) if p is not None]
    labels = [l for p, c, l in zip(predictions, confidences, labels) if p is not None]

    acc, f1, precision, recall = eval_metric(predictions, labels)
    print(acc, f1, precision, recall)
    import pdb; pdb.set_trace()