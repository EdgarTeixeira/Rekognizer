import json
from pathlib import Path

from pandas import read_csv


def filter_hard(rows):
    bboxes = rows['bboxes']
    img_size = rows['img_size']
    return [
        bbox
        for bbox in bboxes
        if bbox['occlusion'] in {0, 1}
        and bbox['pose'] == 0
        and bbox['invalid'] == 0
        and (bbox['w'] * bbox['h']) / img_size >= .005
    ]


if __name__ == '__main__':
    project_root = Path(__file__).absolute().parent.parent
    labels_df = read_csv(project_root/'data/face_detection/labels.csv')\
        .assign(
            bboxes=lambda df: df.bboxes.apply(json.loads),
            img_size=lambda df: df.img_height * df.img_width)

    labels_df.bboxes = labels_df.apply(filter_hard,
                                       axis=1,
                                       result_type='reduce')

    filepath = project_root/'out/viola_jones/background.txt'
    filepath.parent.mkdir(exist_ok=True)
    with open(filepath, mode='w', encoding='utf-8') as file:
        for row in labels_df.itertuples():
            if len(row.bboxes) == 0:
                file.write(f'imgs/{row.filename}\n')

    filepath = project_root/'out/viola_jones/objects.txt'
    filepath.parent.mkdir(exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as file:
        for row in labels_df.itertuples():
            if len(row.bboxes) != 0:
                line = [f'imgs/{row.filename}', str(len(row.bboxes))]
                for bbox in row.bboxes:
                    line.append(
                        f'{bbox["x"]} {bbox["y"]} {bbox["w"]} {bbox["h"]}')
                line.append('\n')
                file.write('   '.join(line))
