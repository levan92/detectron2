import json
import argparse
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument('preds',help='path to instances_predictions.pth')
args = ap.parse_args()

pred_file = Path(args.preds)
preds = torch.load(pred_file)

all_preds = []
all_cats = []
for pred in preds:
    img_id = pred['image_id']
    instances = pred['instances']
    for instance in instances:
        instance['category_id'] += 1
        all_preds.append(instance)

out_json = pred_file.parent / 'coco_instances_preds.json'
with out_json.open('w') as f:
    json.dump(all_preds, f)

print(f'Outputted to {out_json}!')