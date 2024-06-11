import os
import random
import shutil

train_data_dir = 'data/train'
train_mask_dir = 'data/train_masks'
val_data_dir = 'data/val'
val_mask_dir = 'data/val_masks'

if not os.path.isdir(val_data_dir):
    os.mkdir(val_data_dir)
if not os.path.isdir(val_mask_dir):
    os.mkdir(val_mask_dir)


item_list = os.listdir(train_data_dir)

count = len(item_list)

val_idx = []

for _ in range(int(count * 0.15)):
    val_idx.append(random.randint(0, count - 1))

val_idx = set(val_idx)

for idx in val_idx:
    item = item_list[idx]
    img_path = os.path.join(train_data_dir, item)
    mask_path = os.path.join(train_mask_dir, item)
    mask_path = mask_path.replace('.jpg', '_mask.gif')
    shutil.move(img_path, img_path.replace('train', 'val'))
    shutil.move(mask_path, mask_path.replace('train', 'val'))