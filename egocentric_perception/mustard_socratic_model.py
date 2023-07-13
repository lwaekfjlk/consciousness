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
        prompt = """
Sarcasm is when people don't respond directly but they respond in an unexpected way with a special intention outside of the sentence itself. And this special intention is to show their unwillingness or unsatisfaction.
Not sarcasm is when response doesn't include any intentional information and just want to say what the response means. And the response does not carry any unwillingness or unsatisfaction.
        """
        prompt += """
For example,
SHELDON: And I cannot believe people pay for horoscopes, but on a more serious note, it's 8:13 and we're still not playing Halo.
LEONARD: Fine. We'll just play one-on-one until he gets back.
SHELDON: One-on-one? We don't play one-on-one.
SHELDON: We play teams, not one-on-one.
SHELDON: One-on-one.
LEONARD: The only way we can play teams at this point is if we cut Raj in half.
The utterance said by RAJ "Sure, cut the foreigner in half. There's a billion more where he came from." is sarcasm because it shows the unwillingness of RAJ to be cut in half.

For example,
LEONARD: Do you really need the Honorary Justice League of American membership card?
SHELDON: It's been in every wallet I've owned since I was five.
LEONARD: Why?
SHELDON: It says, \"Keep this on your person at all times.\"
SHELDON: It's right here under Batman's signature.
RAJ: and this is Leonard and Sheldon's apartment.
HOWARD: Guess whose parents just got broadband.
RAG: Leonard, may I present, live from New Delhi, Dr. and Mrs. V. M. Koothrappali.
The utterance said by RAJ "Leonard, may I present, live from New Delhi, Dr. and Mrs. V. M. Koothrappali." is not sarcasm because it is just a normal introduction.
        """
        prompt += """
Here is the final question:\n
        """
        utterance = data['utterance']
        context = data['context']
        utterance_speaker = data['speaker']
        context_speaker = data['context_speakers']
        for s, u in zip(context_speaker, context):
            prompt += '{}: {}\n'.format(s, u)
        prompt += '{}: {}\n'.format(utterance_speaker, utterance)
        #prompt += f'Question: Is the last utterance sarcastic? Rate your confidence for your answer from 1-5 and answer with YES or NO and explain the reason for your judgement: ' 
        prompt += f'Question: Is the last utterance sarcastic? Answer with YES or NO:'
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
        prompt += f'Question: Is the last utterance sarcastic? Answer with YES or NO: ' 
        prompts.append(prompt)
    return prompts


def build_vision_prompt(dataset, vision_info):
    prompts = []
    for index, (idx, data) in enumerate(tqdm(dataset.items())):
        utterance = data['utterance']
        if len(vision_info[idx]) == 0:
            vision_emotion = 'neutral'
        else:
            vision_emotion = ', '.join(vision_info[idx])
        prompt = "Think about the relationship between facial expressions like happy / angry / sad / neutral and whether a sentence is sarcastic or not.\n"
        prompt += f"The last utterance '{utterance}' is said with {vision_emotion} face.\n"
        prompt += f'Question: Is the last utterance sarcastic? Rate your confidence for your answer from 1-5 and answer with YES or NO: ' 
        prompts.append(prompt)
    return prompts


def generate(prompts):
    # config for generation
    model_name = 'gpt-3.5-turbo-0613'
    max_output_len = 512
    temp = 0.2
    p = 0.8
    config = lm_config.LMConfig(provider='openai', model=model_name)

    # Initialize lists with None
    predictions = [None for _ in range(len(prompts))]
    confidences = [None for _ in range(len(prompts))]

    # Run the loop until all predictions are filled
    iter_num = 0
    while iter_num <= 50:
        iter_num += 1
        gen_ans = []
        for i in range(0, len(prompts), 10):
            while True:
                try:
                    gen_ans += asyncio.run(asyncio.wait_for(generate_from_openai_chat_completion(
                        prompts[i:i+10],
                        config,
                        temperature=temp,
                        max_tokens=max_output_len,
                        top_p=p,
                        requests_per_minute=100,
                    ), timeout=10))
                    break
                except:
                    print('Failed! Sleep 10 seconds and try again.')
                    time.sleep(10)

        new_prompts = []
        new_indices = []
        for idx, ans in enumerate(gen_ans):
            prediction, confidence = filter_invalid_ans(ans)
            if prediction is None:
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

    predictions, confidences = generate(audio_prompts)

    #predictions = torch.load('./lang_only_predictions_few_shot.pt')
    #confidences = torch.load('./lang_only_confidences_few_shot.pt')
    #labels = torch.load('./lang_only_labels_few_shot.pt')
    for idx, (p, c, l) in enumerate(zip(predictions, confidences, labels)):
        if p is None:
            predictions[idx] = False
            confidences[idx] = 5

    predictions_ = [p for p, c, l in zip(predictions, confidences, labels) if p is not None]
    confidences_ = [c for p, c, l in zip(predictions, confidences, labels) if p is not None]
    labels_ = [l for p, c, l in zip(predictions, confidences, labels) if p is not None]

    acc, f1, precision, recall = eval_metric(predictions_, labels_)
    print(acc, f1, precision, recall)
    #torch.save(predictions, './vision_only_predictions.pt')
    #torch.save(confidences, './vision_only_confidences.pt')
    #torch.save(labels, './vision_only_labels.pt')
    import pdb; pdb.set_trace()