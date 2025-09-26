import os
import shutil
import random
random.seed(309)
from typing import List, Dict
from collections import defaultdict


def reset_directory(path: str) -> None:
    """Remove directory if it exists so we can recreate a clean tree."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

BLACK = '\033[30m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
RESET = '\033[0m'

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

"""
names:
  0: body
  1: adult
  2: child
  3: male
  4: female
  5: body_with_wheelchair
  6: body_with_crutches
  7: head
  8: front
  9: right-front
  10: right-side
  11: right-back
  12: back
  13: left-back
  14: left-side
  15: left-front
  16: face
  17: eye
  18: nose
  19: mouth
  20: ear
  21: collarbone
  22: shoulder
  23: solar_plexus
  24: elbow
  25: wrist
  26: hand
  27: hand_left
  28: hand_right
  29: abdomen
  30: hip_joint
  31: knee
  32: ankle
  33: foot
"""

# CVATから抽出したラベルリストをそのまま貼り付け
LABELS: List[Dict] = \
[
  {
    "name": "body",
    "id": 488,
    "color": "#3d3df5",
    "type": "any",
    "attributes": []
  },
  {
    "name": "adult",
    "id": 489,
    "color": "#34d1b7",
    "type": "any",
    "attributes": []
  },
  {
    "name": "child",
    "id": 490,
    "color": "#ff00cc",
    "type": "any",
    "attributes": []
  },
  {
    "name": "male",
    "id": 491,
    "color": "#32b7fa",
    "type": "any",
    "attributes": []
  },
  {
    "name": "female",
    "id": 492,
    "color": "#ff6037",
    "type": "any",
    "attributes": []
  },
  {
    "name": "body_with_wheelchair",
    "id": 493,
    "color": "#b83df5",
    "type": "any",
    "attributes": []
  },
  {
    "name": "body_with_crutches",
    "id": 494,
    "color": "#24b353",
    "type": "any",
    "attributes": []
  },
  {
    "name": "head",
    "id": 495,
    "color": "#fa3253",
    "type": "any",
    "attributes": []
  },
  {
    "name": "front",
    "id": 496,
    "color": "#33ddff",
    "type": "any",
    "attributes": []
  },
  {
    "name": "right-front",
    "id": 497,
    "color": "#fa3253",
    "type": "any",
    "attributes": []
  },
  {
    "name": "right-side",
    "id": 498,
    "color": "#34d1b7",
    "type": "any",
    "attributes": []
  },
  {
    "name": "right-back",
    "id": 499,
    "color": "#ff6037",
    "type": "any",
    "attributes": []
  },
  {
    "name": "back",
    "id": 500,
    "color": "#ddff33",
    "type": "any",
    "attributes": []
  },
  {
    "name": "left-back",
    "id": 501,
    "color": "#24b353",
    "type": "any",
    "attributes": []
  },
  {
    "name": "left-side",
    "id": 502,
    "color": "#b83df5",
    "type": "any",
    "attributes": []
  },
  {
    "name": "left-front",
    "id": 503,
    "color": "#66ff66",
    "type": "any",
    "attributes": []
  },
  {
    "name": "face",
    "id": 504,
    "color": "#ff6a4d",
    "type": "any",
    "attributes": []
  },
  {
    "name": "eye",
    "id": 505,
    "color": "#fa32b7",
    "type": "any",
    "attributes": []
  },
  {
    "name": "nose",
    "id": 506,
    "color": "#3df53d",
    "type": "any",
    "attributes": []
  },
  {
    "name": "mouth",
    "id": 507,
    "color": "#33ddff",
    "type": "any",
    "attributes": []
  },
  {
    "name": "ear",
    "id": 508,
    "color": "#fafa37",
    "type": "any",
    "attributes": []
  },
  {
    "name": "collarbone",
    "id": 509,
    "color": "#fa3253",
    "type": "any",
    "attributes": []
  },
  {
    "name": "shoulder",
    "id": 510,
    "color": "#fafa37",
    "type": "any",
    "attributes": []
  },
  {
    "name": "solar_plexus",
    "id": 511,
    "color": "#ddff33",
    "type": "any",
    "attributes": []
  },
  {
    "name": "elbow",
    "id": 512,
    "color": "#ddff33",
    "type": "any",
    "attributes": []
  },
  {
    "name": "wrist",
    "id": 513,
    "color": "#ddff33",
    "type": "any",
    "attributes": []
  },
  {
    "name": "hand",
    "id": 514,
    "color": "#66ff66",
    "type": "any",
    "attributes": []
  },
  {
    "name": "hand_left",
    "id": 515,
    "color": "#cc9933",
    "type": "any",
    "attributes": []
  },
  {
    "name": "hand_right",
    "id": 516,
    "color": "#aaf0d1",
    "type": "any",
    "attributes": []
  },
  {
    "name": "abdomen",
    "id": 517,
    "color": "#3df53d",
    "type": "any",
    "attributes": []
  },
  {
    "name": "hip_joint",
    "id": 518,
    "color": "#3df53d",
    "type": "any",
    "attributes": []
  },
  {
    "name": "knee",
    "id": 519,
    "color": "#3df53d",
    "type": "any",
    "attributes": []
  },
  {
    "name": "ankle",
    "id": 520,
    "color": "#3df53d",
    "type": "any",
    "attributes": []
  },
  {
    "name": "foot",
    "id": 521,
    "color": "#733380",
    "type": "any",
    "attributes": []
  }
]

def split_dataset(base_dir, output_dir, split_ratio=(0.8, 0.2)):
    train_image_dir = os.path.join(output_dir, 'images', 'train')
    val_image_dir = os.path.join(output_dir, 'images', 'val')
    train_label_dir = os.path.join(output_dir, 'labels', 'train')
    val_label_dir = os.path.join(output_dir, 'labels', 'val')

    # ディレクトリの作成
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # 画像ファイルとアノテーションファイルのリストを取得
    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]

    # 画像ファイルとアノテーションファイルを対応付ける
    dataset = []
    for file in files:
        if file.lower().endswith(('.jpg', '.png')):  # 画像ファイルの拡張子をチェック
            label_file = file.rsplit('.', 1)[0] + '.txt'  # 対応するラベルファイル名を作成
            if label_file in files:
                dataset.append((file, label_file))

    # データセットをシャッフル
    random.shuffle(dataset)

    # 分割位置を計算
    split_point = int(len(dataset) * split_ratio[0])
    train_set = dataset[:split_point]
    val_set = dataset[split_point:]

    # 各セットのclass_idをカウントする辞書
    train_class_id_count = defaultdict(int)
    val_class_id_count = defaultdict(int)

    def move_files(file_set, img_dest, label_dest, class_id_count):
        for image_file, label_file in file_set:
            # 画像ファイルの移動
            shutil.copy(os.path.join(base_dir, image_file), os.path.join(img_dest, image_file))
            # アノテーションファイルの移動
            shutil.copy(os.path.join(base_dir, label_file), os.path.join(label_dest, label_file))

            # Unknown ラベルの削除
            with open(os.path.join(label_dest, label_file), "r") as file:
                lines = file.readlines()
            filtered_lines = [line for line in lines]
            with open(os.path.join(label_dest, label_file), "w") as file:
                file.writelines(filtered_lines)

            # class_id のカウント
            for line in filtered_lines:
                try:
                  class_id = int(line.split()[0])
                  class_id_count[class_id] += 1
                except:
                  pass

    # ファイルを移動し、class_id をカウント
    move_files(train_set, train_image_dir, train_label_dir, train_class_id_count)
    move_files(val_set, val_image_dir, val_label_dir, val_class_id_count)

    # 画像の総枚数とtrain/valごとの画像数を表示
    total_images = len(dataset)
    train_images = len(train_set)
    val_images = len(val_set)
    print('')
    print(f'{YELLOW}Train images     :{RESET}{train_images:>6,}')
    print(f'{YELLOW}Validation images:{RESET}{val_images:>6,}')
    print(f'{GREEN}Total images     :{RESET}{total_images:>6,}')
    print('===================================================')

    # train セットの class_id のカウント結果を表示
    print(f'{RED}Train Set Class ID Count{RESET}')
    total_train_count = 0
    for class_id, count in sorted(train_class_id_count.items()):
        print(f'{BLUE}class_id:{RESET}{int(class_id):>2} {BLUE}name:{RESET}{LABELS[int(class_id)].get("name", "unknown"):>20} {BLUE}count:{RESET}{count:>7,}')
        total_train_count += count
    print('---------------------------------------------------')
    print(f'{GREEN}Total count for train set:{RESET}{total_train_count:>7,}')
    print('===================================================')

    # val セットの class_id のカウント結果を表示
    print(f'{RED}Validation Set Class ID Count{RESET}')
    total_val_count = 0
    for class_id, count in sorted(val_class_id_count.items()):
        print(f'{BLUE}class_id:{RESET}{int(class_id):>2} {BLUE}name:{RESET}{LABELS[int(class_id)].get("name", "unknown"):>20} {BLUE}count:{RESET}{count:>7,}')
        total_val_count += count
    print('---------------------------------------------------')
    print(f'{GREEN}Total count for validation set:{RESET}{total_val_count:>7,}')
    print('===================================================')
    print('')

def list_image_files(directory: str) -> List[str]:
    return [
        entry
        for entry in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, entry))
        and entry.lower().endswith(IMAGE_EXTENSIONS)
    ]


def copy_files(src_dir: str, dest_dir: str, filenames: List[str]) -> None:
    for name in filenames:
        src_path = os.path.join(src_dir, name)
        if not os.path.isfile(src_path):
            raise FileNotFoundError(f'Missing source file: {src_path}')
        shutil.copy(src_path, os.path.join(dest_dir, name))


def create_swapped_dataset(src_dataset_dir: str, dest_dataset_dir: str) -> None:
    images_train = os.path.join(src_dataset_dir, 'images', 'train')
    images_val = os.path.join(src_dataset_dir, 'images', 'val')
    labels_train = os.path.join(src_dataset_dir, 'labels', 'train')
    labels_val = os.path.join(src_dataset_dir, 'labels', 'val')

    for required in (images_train, images_val, labels_train, labels_val):
        if not os.path.isdir(required):
            raise FileNotFoundError(f'Missing source directory: {required}')

    val_images = sorted(list_image_files(images_val))
    train_images = sorted(list_image_files(images_train))

    swap_count = len(val_images)
    if swap_count == 0:
        raise ValueError('dataset1 val split is empty; nothing to swap.')
    if len(train_images) < swap_count:
        raise ValueError('Not enough dataset1 train samples to swap with val set.')

    train_swap = random.sample(train_images, swap_count)
    train_swap_set = set(train_swap)
    train_remaining = [name for name in train_images if name not in train_swap_set]

    reset_directory(dest_dataset_dir)

    images_train_dest = os.path.join(dest_dataset_dir, 'images', 'train')
    images_val_dest = os.path.join(dest_dataset_dir, 'images', 'val')
    labels_train_dest = os.path.join(dest_dataset_dir, 'labels', 'train')
    labels_val_dest = os.path.join(dest_dataset_dir, 'labels', 'val')

    target_structure = [
        images_train_dest,
        images_val_dest,
        labels_train_dest,
        labels_val_dest,
    ]

    for directory in target_structure:
        os.makedirs(directory, exist_ok=True)

    copy_files(images_train, images_train_dest, train_remaining)
    copy_files(images_val, images_train_dest, val_images)
    copy_files(images_train, images_val_dest, train_swap)

    def labels_for(image_names: List[str]) -> List[str]:
        return [os.path.splitext(name)[0] + '.txt' for name in image_names]

    copy_files(labels_train, labels_train_dest, labels_for(train_remaining))
    copy_files(labels_val, labels_train_dest, labels_for(val_images))
    copy_files(labels_train, labels_val_dest, labels_for(train_swap))

    print('')
    print(f'{YELLOW}Created swapped dataset at:{RESET} {dest_dataset_dir}')
    print(f'{YELLOW}dataset2/train:{RESET} {len(train_remaining) + len(val_images):>6,} (kept train {len(train_remaining):>6,} + val {len(val_images):>6,})')
    print(f'{YELLOW}dataset2/val  :{RESET} {len(train_swap):>6,} (sampled from train)')

def main() -> None:
    base_directory = 'wholebody34/obj_train_data'
    dataset1_dir = 'dataset1'
    dataset2_dir = 'dataset2'

    reset_directory(dataset1_dir)
    split_dataset(base_directory, dataset1_dir)
    create_swapped_dataset(dataset1_dir, dataset2_dir)


if __name__ == '__main__':
    main()
