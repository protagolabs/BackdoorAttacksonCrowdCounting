import json
from os.path import join
import glob


if __name__ == '__main__':
    # path to folder that contains images
    img_folder = './datasets/poinson/part_B_final/train_data/images'
    # img_folder = './datasets/poinson/part_B_final/test_data/images'
    # img_folder= './datasets/clean/part_B_final/train_data/images'
    img_folder = "./datasets/cleanpart_B_final/test_data/images"

    # path to the final json file
    output_json = './part_B_train_portion0.2.json'
    # output_json = './part_B_test_portion0.2.json'
    # output_json = './part_B_train.json'
    # output_json = './part_B_test.json'

    img_list = []

    for img_path in glob.glob(join(img_folder,'*.jpg')):
        img_list.append(img_path)

    with open(output_json,'w') as f:
        json.dump(img_list,f)
