from datasets import load_dataset
import jsonlines
import json

dataset = list(load_dataset("jmhessel/newyorker_caption_contest", "matching", split='test'))
explain_test_dataset = list(load_dataset("jmhessel/newyorker_caption_contest", "explanation", split='test'))
explain_train_dataset = list(load_dataset("jmhessel/newyorker_caption_contest", "explanation", split='train'))
explain_validation_dataset = list(load_dataset("jmhessel/newyorker_caption_contest", "explanation", split='validation'))
explain_dataset = explain_test_dataset + explain_train_dataset + explain_validation_dataset

explain_context = {}
for data in explain_dataset:
    explain_context[data['instance_id']] = data

context = {}
for data in dataset:
    context[data['instance_id']] = data

labels = {}
for data in dataset:
    labels[data['instance_id']] = data['label']

with jsonlines.open('./gpt4/query_cache_answer_gpt3.5/matching_gpt-3.5-turbo_5_cache.jsonl', 'r') as f:
    predictions = [d for d in f]

preds = {}
for pred in predictions:
    preds[pred['instance_id']] = pred['response'].replace('Answer: ', '')


correct = 0
for instance_id, pred in preds.items():
    label = labels[instance_id]
    if pred == label:
        correct += 1
        print(context[instance_id]['image_description'])
        print(context[instance_id]['questions'])
        for choice in context[instance_id]['caption_choices']:
            print(choice)
        print("pred label: {}".format(pred))
        print("gth label: {}".format(label))
        if instance_id in explain_context.keys():
            print("explanation: {}".format(explain_context[instance_id]['explanation']))
        import pdb; pdb.set_trace()

print(correct / len(preds))
