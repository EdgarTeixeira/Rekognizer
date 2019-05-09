import json
from pathlib import Path

from pandas import DataFrame

from cv2 import imread


if __name__ == '__main__':
    project_root = Path(__file__).absolute().parent.parent
    data_dir = project_root/"data/face_detection/WIDER_train/images"
    label_dir = project_root/"data/face_detection/wider_face_split/wider_face_train_bbx_gt.txt"
    label_names = [
        'x',
        'y',
        'w',
        'h',
        'blur',
        'expression',
        'illumination',
        'invalid',
        'occlusion',
        'pose'
    ]
    bounding_boxes = []

    with open(label_dir, 'r') as file:
        next_line = file.readline().strip()
        while next_line != '':
            filename = next_line
            img_shape = imread(str(data_dir/filename)).shape
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

    data_df = DataFrame.from_records(bounding_boxes)
    data_df.to_csv(project_root/'data/face_detection/labels.csv', index=False)
