import datetime
import json
import os
import re
import time

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

gpt_version = "text-davinci-003"
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
    gen_res = [prompt_llm(prompt, temperature=0.9) for _ in range(activity_topk)]
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


if __name__ == '__main__':
    clip_version = "ViT-L/14"
    clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768, 'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
    model, preprocess = clip.load(clip_version)
    model.cuda().eval()

    img_size = model.visual.input_resolution
    fname = '2_448_frame.png'
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    state_world = get_state_world(img)

    prompt = f"""
{state_world}
They say "Do you still wanna call  'em? I wanna call  'em."
Question: Is this sarcasm? Answer and Explain. Answer: """

    socratic_res = prompt_llm(prompt, temperature=0.9)
    print(socratic_res)
    