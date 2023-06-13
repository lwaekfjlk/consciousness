import datetime
import json
import os
import re
import time

import requests
import cv2
import matplotlib.pyplot as plt
import numpy as np
import openai
import clip
from PIL import Image
from profanity_filter import ProfanityFilter
import torch


def get_text_feats(in_text, batch_size=64):
    text_tokens = clip.tokenize(in_text).cuda()
    text_id = 0
    text_feats = np.zeros((len(in_text), clip_feat_dim), dtype=np.float32)
    while text_id < len(text_tokens):  # Batched inference.
        batch_size = min(len(in_text) - text_id, batch_size)
        text_batch = text_tokens[text_id:text_id+batch_size]
        with torch.no_grad():
            batch_feats = model.encode_text(text_batch).float()
            batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
            batch_feats = np.float32(batch_feats.cpu())
            text_feats[text_id:text_id+batch_size, :] = batch_feats
            text_id += batch_size
    return text_feats


def get_img_feats(img):
    img_pil = Image.fromarray(np.uint8(img))
    img_in = preprocess(img_pil)[None, ...]
    with torch.no_grad():
        img_feats = model.encode_image(img_in.cuda()).float()
    img_feats /= img_feats.norm(dim=-1, keepdim=True)
    img_feats = np.float32(img_feats.cpu())
    return img_feats


def get_nn_text(raw_texts, text_feats, img_feats):
    scores = text_feats @ img_feats.T
    scores = scores.squeeze()
    high_to_low_ids = np.argsort(scores).squeeze()[::-1]
    high_to_low_texts = [raw_texts[i] for i in high_to_low_ids]
    high_to_low_scores = np.sort(scores).squeeze()[::-1]
    return high_to_low_texts, high_to_low_scores


def prompt_llm(prompt, max_tokens=64, temperature=0, stop=None):
    response = openai.Completion.create(engine=gpt_version, prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
    return response["choices"][0]["text"].strip()


def load_texts():
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
    for object_text in object_categories[1:]:
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
    object_feats = get_text_feats([f'Photo of a {o}.' for o in object_texts])
    return place_texts, object_texts


def img_summary(img):
    verbose = True #@param {type:"boolean"}
    img_feats = get_img_feats(img)
    plt.imshow(img); plt.show()

    place_texts, object_texts = load_texts()
    # Zero-shot VLM: classify places.
    place_topk = 3
    place_feats = get_text_feats([f'Photo of a {p}.' for p in place_texts ])
    sorted_places, places_scores = get_nn_text(place_texts, place_feats, img_feats)

    # Zero-shot VLM: classify objects.
    obj_topk = 3
    object_feats = get_text_feats([f'Photo of a {o}.' for o in object_texts])
    sorted_obj_texts, obj_scores = get_nn_text(object_texts, object_feats, img_feats)
    object_list = ''
    for i in range(obj_topk):
        object_list += f'{sorted_obj_texts[i]}, '
        object_list = object_list[:-2]

    '''
    place_topk = 3
    obj_topk = 3
    sorted_places = ['staircase', 'house', 'bedroom']
    sorted_obj_texts = ['step, stair', 'stair-carpet', 'lesser ape']
    '''

    # Zero-shot LM: generate activities.
    num_activity = 1
    activity_texts = []
    prompt = f'''Places: {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.
    Objects: {sorted_obj_texts[0]}, {sorted_obj_texts[1]}, {sorted_obj_texts[2]}.
    Activities (separate them with /): '''
    gen_res = [prompt_llm(prompt, temperature=0.9) for _ in range(num_activity)]
    for activity in gen_res:
        activities = [a.strip().replace('.', '') for a in activity.split('/')]
        activity_texts += activities

    # Zero-shot VLM: rank activities.
    activity_feats = get_text_feats(activity_texts)
    sorted_activities, activity_scores = get_nn_text(activity_texts, activity_feats, img_feats)

    # Zero-shot LM: generate image summary.
    prompt = f'''I am in a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.
    I see a {sorted_obj_texts[0]}, {sorted_obj_texts[1]}, {sorted_obj_texts[2]}.
    I am {sorted_activities[0]}, {sorted_activities[1]}, or {sorted_activities[2]}.
    Question: What am I doing? Answer: I am most likely '''
    img_summary = prompt_llm(prompt, temperature=0.9)


    state_world_history = f'''I am in a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.
    I see a {sorted_obj_texts[0]}, {sorted_obj_texts[1]}, {sorted_obj_texts[2]}.
    I am {sorted_activities[0]}, {sorted_activities[1]}, or {sorted_activities[2]}.
    I am most likely {img_summary}
    ''' 

    if verbose:
        print(state_world_history)

    return state_world_history


if __name__ == '__main__':
    openai_api_key = os.environ['OPENAI_API_KEY']
    openai.api_key = openai_api_key

    clip_version = "ViT-L/14" #@param ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14"] {type:"string"}
    gpt_version = "text-davinci-003" #@param ["text-davinci-001", "text-davinci-002", "text-curie-001", "text-babbage-001", "text-ada-001"] {type:"string"}

    clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768, 'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]

    # torch.cuda.set_per_process_memory_fraction(0.9, None)  # Only needed if session crashes.
    model, preprocess = clip.load(clip_version)  # clip.available_models()
    model.cuda().eval()

    def num_params(model):
        return np.sum([int(np.prod(p.shape)) for p in model.parameters()])
    print("Model parameters (total):", num_params(model))
    print("Model parameters (image encoder):", num_params(model.visual))
    print("Model parameters (text encoder):", num_params(model.token_embedding) + num_params(model.transformer))
    print("Input image resolution:", model.visual.input_resolution)
    print("Context length:", model.context_length)
    print("Vocab size:", model.vocab_size)
    img_size = model.visual.input_resolution

    '''
    state_world_histories = []
    fname = 'frame_34144.png'
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    state_world_histories.append(img_summary(img))
    import pdb; pdb.set_trace()
    '''

    prompt = """
I am in a pet shop, candy store, or fastfood restaurant.
I see a employee, money handler, money dealer, salesman.
I am Processing payments, Handling money, or Greeting customers.
I am most likely processing payments or handling money.
Question: Where did the man put the money ? Answer: 
    """

    socratic_res = prompt_llm(prompt, temperature=0.9)
    print(socratic_res)
    