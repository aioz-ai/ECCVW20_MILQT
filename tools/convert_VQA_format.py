import json

def convert_VQA_format(split = 'train'):
    if split == 'train':
        data = json.load(open('mscoco_train2014_annotations.json'))
    elif split == 'val':
        data = json.load(open('mscoco_val2014_annotations.json'))
    result = {}
    annotations = []
    for d in data['annotations']:
        d['multiple_choice_answer'] = d['answers'][0]['answer']
        d['answers'] = 10*d['answers']
        annotations.append(d)
    result['annotations'] = annotations
    if split == 'train':
        with open('mscoco_train2014_annotations_converted.json','w') as f:
            json.dump(result,f)
    if split == 'val':
        with open('mscoco_val2014_annotations_converted.json','w') as f:
            json.dump(result,f)

convert_VQA_format('train')
convert_VQA_format('val')