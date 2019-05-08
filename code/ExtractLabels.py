#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pathlib
import pandas as pd
import cv2


# In[2]:


data_dir = pathlib.Path("../../repos/datasets/Face Detection/WIDER_train/images")
label_dir = pathlib.Path("../../repos/datasets/Face Detection/wider_face_split/wider_face_train_bbx_gt.txt")
label_names = ['x', 'y', 'w', 'h', 'blur', 'expression', 'illumination', 'invalid', 'occlusion', 'pose']
bounding_boxes = []

with open(label_dir, 'r') as file:
    next_line = file.readline().strip()
    while next_line != '':
        filename = next_line
        img_shape = cv2.imread(str(data_dir/filename)).shape
        num_faces = int(file.readline())
        bboxes = [None for _ in range(num_faces)]

        if num_faces == 0:
            _ = file.readline()
        
        for idx in range(num_faces):
            labels = map(int, file.readline().split(' '))
            bboxes[idx] = dict(zip(label_names, labels))

        bounding_boxes.append({
            'filename': filename,
            'img_height': img_shape[0],
            'img_width': img_shape[1],
            'bboxes': json.dumps(bboxes)
        })
        
        next_line = file.readline().strip()


# In[3]:


data_df = pd.DataFrame.from_records(bounding_boxes)
data_df.to_csv('labels.csv', index=False)

