import os
import shutil

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
  21: collarbone # 0->21
  22: shoulder # 21->22
  23: solar_plexus # 1->23
  24: elbow # 22->24
  25: wrist # 2->25
  26: hand # 23->26
  27: hand_left # 24->27
  28: hand_right # 25->28
  29: abdomen # 3->29
  30: hip_joint # 4->30
  31: knee # 26->31
  32: ankle # 5->32
  33: foot # 27->33
"""

# =========================================================
# 0. フォルダのパス設定
# =========================================================
folder34 = "wholebody34/obj_train_data"
folder28 = "wholebody28/obj_train_data"
folder34_k = "wholebody34_k/obj_train_data"
folder34_l = "wholebody34_l/obj_train_data"
images_folder = "images"
images_dist_folder = "images_dist"

# =========================================================
# 1. wholebody34/obj_train_data の .txt のクラスIDを以下の基準で更新する
#    0->21, 1->23, 2->25, 3->29, 4->30, 5->32
# =========================================================
map34 = {
    0: 21,
    1: 23,
    2: 25,
    3: 29,
    4: 30,
    5: 32
}

def update_class_ids_in_folder_34():
    for filename in os.listdir(folder34):
        if not filename.lower().endswith(".txt"):
            continue

        txt_path = os.path.join(folder34, filename)
        new_lines = []

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                old_class_id = int(parts[0])
                if old_class_id in map34:
                    parts[0] = str(map34[old_class_id])
                new_lines.append(" ".join(parts))

        # 上書き保存
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")

# =========================================================
# 2. wholebody28/obj_train_data の .txt のクラスIDを指定基準で変更し、
#    同名の .txt が wholebody34/obj_train_data に存在する場合のみ追記する
#
#    ※ 質問文では "22->24, 22->26" とあり矛盾がありますが、ここでは以下のように仮定:
#      21->22, 22->24, 23->26, 24->27, 25->28, 26->31, 27->33
#    ※ もし実際のマッピングが異なる場合は下記 map28 を修正してください。
# =========================================================
map28 = {
    21: 22,
    22: 24,
    23: 26,
    24: 27,
    25: 28,
    26: 31,
    27: 33
}

def append_changed_class_ids_from_folder_28():
    for filename in os.listdir(folder28):
        if not filename.lower().endswith(".txt"):
            continue

        # wholebody34 に同名txtがない場合は処理しない
        txt_path_34 = os.path.join(folder34, filename)
        if not os.path.exists(txt_path_34):
            # スキップ
            continue

        txt_path_28 = os.path.join(folder28, filename)

        # 変更後の行を一時的に格納
        modified_lines = []
        with open(txt_path_28, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                old_class_id = int(parts[0])
                # map28 に該当すれば変換
                if old_class_id in map28:
                    parts[0] = str(map28[old_class_id])
                modified_lines.append(" ".join(parts))

        # wholebody34/obj_train_data の同名ファイルに追記
        with open(txt_path_34, "a", encoding="utf-8") as f:
            for ml in modified_lines:
                f.write(ml + "\n")

# =========================================================
# 3. wholebody34/obj_train_data に存在する .txt と拡張子を除く部分の
#    ファイル名が一致する images フォルダの画像ファイルをコピー
# =========================================================
image_extensions = [".jpg", ".JPG", ".png", ".PNG"]

def copy_images_from_images_folder():
    # wholebody34/obj_train_data にあるすべての .txt ファイル名(拡張子除く)を取得
    txt_files = [f for f in os.listdir(folder34) if f.lower().endswith(".txt")]
    txt_basenames = [os.path.splitext(f)[0] for f in txt_files]

    for bn in txt_basenames:
        for ext in image_extensions:
            img_name = bn + ext
            src_path = os.path.join(images_folder, img_name)
            if os.path.exists(src_path):
                dst_path = os.path.join(folder34, img_name)
                shutil.copy2(src_path, dst_path)
                # 同名で複数拡張子がある場合、必要に応じて break
                break

# =========================================================
# 4. wholebody34/obj_train_data に存在する .txt のファイル名(拡張子除く)と
#    images_dist フォルダの "dist_" 以降のファイル名が一致するものをコピー。
#    ただし、wholebody34/obj_train_data にその "dist_" 以降のファイル名
#    (＝元となる .txt) が存在しない場合はスキップ。
#
#    例: wholebody34 に "foo.txt" があれば、
#        images_dist の "dist_foo.txt"/"dist_foo.jpg" 等をコピー
# =========================================================
def copy_files_from_images_dist():
    # wholebody34/obj_train_data にあるすべての .txt ファイル名(拡張子除く)を取得
    txt_files = [f for f in os.listdir(folder34) if f.lower().endswith(".txt")]
    txt_basenames = [os.path.splitext(f)[0] for f in txt_files]
    txt_basenames_set = set(txt_basenames)

    # images_dist フォルダ内を走査
    for dist_file in os.listdir(images_dist_folder):
        # 拡張子を含めずにファイル名を取り出し
        dist_name, dist_ext = os.path.splitext(dist_file)
        if not dist_ext.lower() in image_extensions + [".txt"]:
            # 対象拡張子でなければスキップ
            continue

        # "dist_" で始まるかをチェック
        if not dist_name.startswith("dist_"):
            continue

        # dist_ 以降を取り出して比較する
        original_name = dist_name[5:]  # "dist_" の次からが元のベース名
        # wholebody34 に original_name.txt が存在しなければコピーしない
        if original_name not in txt_basenames_set:
            continue

        # ここまできたらコピー対象
        src_path = os.path.join(images_dist_folder, dist_file)
        dst_path = os.path.join(folder34, dist_file)
        shutil.copy2(src_path, dst_path)

# =========================================================
# 5. wholebody34/obj_train_data に存在する dist_ が先頭に付いていない .txt の内容を
#    同フォルダにある dist_ が先頭に付いている同名 .txt に上書き
#
#    例: "foo.txt" の内容を "dist_foo.txt" があればそちらに上書き
# =========================================================
def overwrite_dist_txt():
    all_txt_files = [f for f in os.listdir(folder34) if f.lower().endswith(".txt")]
    txt_set = set(all_txt_files)

    for f in all_txt_files:
        if f.startswith("dist_"):
            # dist_ で始まるファイルは上書き先なのでスキップ
            continue

        # f は dist_ が付いていない .txt
        dist_f = "dist_" + f
        if dist_f in txt_set:
            # dist_f に f の内容を上書きする
            src_path = os.path.join(folder34, f)
            dst_path = os.path.join(folder34, dist_f)
            with open(src_path, "r", encoding="utf-8") as src, \
                 open(dst_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())

# =========================================================
# 6. wholebody34_k/obj_train_data にある .txt と .jpg を
#    wholebody34/obj_train_data へすべてコピー（上書き可）
# =========================================================
def copy_from_folder34_k():
    for filename in os.listdir(folder34_k):
        low = filename.lower()
        if not (low.endswith(".txt") or low.endswith(".jpg")):
            continue
        src = os.path.join(folder34_k, filename)
        dst = os.path.join(folder34,   filename)
        shutil.copy2(src, dst)   # 既存ファイルがあれば上書き

# =========================================================
# 7. wholebody34_l/obj_train_data にある .txt と .jpg を
#    wholebody34/obj_train_data へすべてコピー（上書き可）
# =========================================================
def copy_from_folder34_l():
    for filename in os.listdir(folder34_l):
        low = filename.lower()
        if not (low.endswith(".txt") or low.endswith(".jpg")):
            continue
        src = os.path.join(folder34_l, filename)
        dst = os.path.join(folder34,   filename)
        shutil.copy2(src, dst)   # 既存ファイルがあれば上書き

# =========================================================
# メイン実行部
# =========================================================
def main():
    # Step1
    update_class_ids_in_folder_34()

    # Step2
    append_changed_class_ids_from_folder_28()

    # Step3
    copy_images_from_images_folder()

    # Step4
    copy_files_from_images_dist()

    # Step5
    overwrite_dist_txt()

    # Step6（追加分）── 既存処理がすべて終わったあとに実行
    copy_from_folder34_k()
    copy_from_folder34_l()


if __name__ == "__main__":
    main()
