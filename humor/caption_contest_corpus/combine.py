import jsonlines
import json
from datasets import load_dataset

lang_info = {}
with jsonlines.open('./gpt4/query_cache/matching_gpt-3.5-turbo_5_cache.jsonl', 'r') as f:
    gpt_res = [t for t in f]

    for data in gpt_res:
        response = data['response']
        rank_str = response.split('Rank: ')[-1]
        rank_list = rank_str.split('>')
        rank_dict = {r: 4-idx for idx, r in enumerate(rank_list)}
        lang_info[data['instance_id']] = rank_dict

vision_info = {}
with jsonlines.open('./clip/task=matching~split=5~valacc=0.64962~pad=1~model=ViT-L*14@336px~lr=5e-06~results.json', 'r') as f:
    clip_res = [t for t in f]

    for data in clip_res:
        s = data['logits']
        rank = sorted(range(len(s)), key=lambda k: s[k])
        chars = 'ABCDE'
        rank_dict = {}
        for idx, r in enumerate(rank):
            rank_dict[chars[r]] = idx
        #rank_dict = {c: r for c, r in zip(chars, rank)}
        vision_info[data['instance_id']] = rank_dict

combined_ans = {}
for instance_id in lang_info.keys():
    vision_rank = vision_info[instance_id]
    lang_rank = lang_info[instance_id]
    overall_rank = {}
    for k in vision_rank.keys():
        try:
            overall_rank[k] = -abs(lang_rank[k] - vision_rank[k])
        except:
            overall_rank = vision_rank
            break
    # select the key with max value
    lang_ans = max(lang_rank, key=lang_rank.get)
    vision_ans = max(vision_rank, key=vision_rank.get)
    if lang_ans != vision_ans:
        ans = max(overall_rank, key=overall_rank.get)
        #ans = lang_ans
    else:
        ans = vision_ans
    combined_ans[instance_id] = ans

dataset = list(load_dataset("jmhessel/newyorker_caption_contest", 'matching')['test'])

labels = []
preds = []
for data in dataset:
    instance_id = data['instance_id']
    labels.append(data['label'])
    preds.append(combined_ans[instance_id])

acc = sum([1 if l == p else 0 for l, p in zip(labels, preds)]) / len(labels)
print(acc)
import pdb; pdb.set_trace()