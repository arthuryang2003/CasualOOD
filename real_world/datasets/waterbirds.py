import os
import pandas as pd
import numpy as np
import csv

root_dir = '../data/waterbirds'  # 你的数据根目录
metadata_path = os.path.join(root_dir, 'metadata.csv')

metadata = pd.read_csv(metadata_path)

split_dict = {
    'train': 0,
    'val': 1,
    'test': 2
}

# 切出train/val/test
train_metadata = metadata[metadata['split'] == split_dict['train']]
val_metadata = metadata[metadata['split'] == split_dict['val']]
test_metadata = metadata[metadata['split'] == split_dict['test']]

# == 训练环境：制造偏置
# landbird+land 90% ; landbird+water 10%
# waterbird+water 90% ; waterbird+land 10%
np.random.seed(0)

tr1_rows = []
tr2_rows = []

for idx, row in train_metadata.iterrows():
    y = row['y']          # 0=landbird, 1=waterbird
    place = row['place']  # 0=land, 1=water
    img_filename = row['img_filename']

    rand_val = np.random.rand()

    if y == 0:  # landbird
        if place == 0 and rand_val < 0.9:  # landbird on land 90%
            tr1_rows.append(row)
        elif place == 1 and rand_val < 0.1:  # landbird on water 10%
            tr1_rows.append(row)
    else:  # waterbird
        if place == 1 and rand_val < 0.9:  # waterbird on water 90%
            tr2_rows.append(row)
        elif place == 0 and rand_val < 0.1:  # waterbird on land 10%
            tr2_rows.append(row)

# == 测试环境：均匀取样
# 50% land background, 50% water background, for both classes

te_rows = []

for y in [0, 1]:
    for place in [0, 1]:
        subset = test_metadata[(test_metadata['y'] == y) & (test_metadata['place'] == place)]
        te_rows.append(subset)

te_rows = pd.concat(te_rows).sample(frac=1, random_state=0)  # 打乱一下

# == 保存csv
os.makedirs(os.path.join(root_dir, 'splits'), exist_ok=True)

pd.DataFrame(tr1_rows).to_csv(os.path.join(root_dir, 'splits', 'tr_env1.csv'), index=False)
pd.DataFrame(tr2_rows).to_csv(os.path.join(root_dir, 'splits', 'tr_env2.csv'), index=False)
pd.DataFrame(te_rows).to_csv(os.path.join(root_dir, 'splits', 'te_env.csv'), index=False)

print("[Info] Waterbirds splits generated successfully!")