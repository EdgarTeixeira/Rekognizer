#!/usr/bin/env python
# coding: utf-8

# In[9]:


import json
from pathlib import Path

import numpy as np
import pandas as pd


# In[3]:


labels_df = pd.read_csv('labels.csv')              .assign(bboxes=lambda df: df.bboxes.apply(json.loads), img_size=lambda df: df.img_height * df.img_width)
labels_df.head()


# In[4]:


def filter_hard(rows):
    bboxes = rows['bboxes']
    img_size = rows['img_size']
    return [
        bbox 
        for bbox in bboxes 
        if bbox['occlusion'] in {0, 1}\
            and bbox['pose'] == 0\
            and bbox['invalid'] == 0
            and (bbox['w'] * bbox['h']) / img_size >= .005
    ]

labels_df.bboxes = labels_df.apply(filter_hard, axis=1, result_type='reduce')


# In[11]:


with open(Path('opencv-data/background.txt'), 'w', encoding='utf-8') as file:
    for row in labels_df.itertuples():
        if len(row.bboxes) == 0:
            file.write(f'imgs/{row.filename}\n')


# In[14]:


with open(Path('opencv-data/objects.txt'), 'w', encoding='utf-8') as file:
    for row in labels_df.itertuples():
        if len(row.bboxes) != 0:
            line = [f'imgs/{row.filename}', str(len(row.bboxes))]
            for bbox in row.bboxes:
                line.append(f'{bbox["x"]} {bbox["y"]} {bbox["w"]} {bbox["h"]}')
            line.append('\n')
            file.write('   '.join(line))


# In[16]:


print("""opencv_createsamples -vec out/faces_w128_h128.vec -info objects.txt -w 128 -h 128""")


# In[1]:


print("opencv_traincascade -data out -vec out\faces_w64_h64.vec -bg background.txt -numPos 64 -numNeg 128 -numStages 5 -precalcValBufSize 2000 -precalcIdxBufSize 2000 -acceptanceRatioBreakValue 10e-5 -w 64 -h 64")

