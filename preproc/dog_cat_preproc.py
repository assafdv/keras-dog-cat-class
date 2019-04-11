'''
This code follows the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

import os
import pandas as pd
from shutil import copyfile
from utils.file_sys_utils import get_files_recur


def preproc(workpath, img_dir):
    """
    pre-process dog cat dataset.
    """

    os.mkdir(os.path.join(workpath, 'subset_1k'))
    # create train dir
    train_dir = os.path.join(workpath, 'subset_1k', 'train')
    os.mkdir(train_dir)
    cats_train_dir = os.path.join(train_dir, 'cats')
    os.mkdir(cats_train_dir)
    dogs_train_dir = os.path.join(train_dir, 'dogs')
    os.mkdir(dogs_train_dir)

    # create validation dir
    val_dir = os.path.join(workpath, 'subset_1k', 'validation')
    os.mkdir(val_dir)
    cats_val_dir = os.path.join(val_dir, 'cats')
    os.mkdir(cats_val_dir)
    dogs_val_dir = os.path.join(val_dir, 'dogs')
    os.mkdir(dogs_val_dir)

    # get cats and dogs images
    cat_imgs = sort_img_file_list(get_img_files('cat', img_dir))
    dog_imgs = sort_img_file_list(get_img_files('dog', img_dir))

    # put the cat pictures index 0-999 in train/cats
    n_train_c = 1000  # no. train samples per class
    n_val_c = 400     # no. validation sampler per class
    copy_img_files(cat_imgs[:n_train_c], img_dir, cats_train_dir)

    # put the cat pictures index 1000-1400 in validation/cats
    copy_img_files(cat_imgs[n_train_c:(n_train_c+n_val_c)], img_dir, cats_val_dir)

    # put the dogs pictures index 12500-13499 in train/dogs
    copy_img_files(dog_imgs[:n_train_c], img_dir, dogs_train_dir)

    # put the dog pictures index 13500-13900 in validation/dogs
    copy_img_files(dog_imgs[n_train_c:(n_train_c+n_val_c)], img_dir, dogs_val_dir)


def get_img_files(category, img_dir):
    """
    return (sorted) list of img files.
    :param category: string, category name (i.e. 'dog' or 'cat').
    :return: list of strings.
    """
    img_files = [file for file in os.listdir(img_dir) if file.startswith(category)]
    return img_files


def sort_img_file_list(file_list):
    """

    :param file_list:
    :return:
    """
    def sort_rule(str_val):
        num = str_val.split('.')[1]
        return int(num)

    return sorted(file_list, key=sort_rule)


def copy_img_files(src_list, src_dir, dst_dir):
    """
    copy list of file to a destination folder
    :param src_list: list (string), list of file names.
    :param src_dir: string, source dir
    :param dst_dir: string, destination dir
    """
    for file in src_list:
        src_path = os.path.join(src_dir, file)
        dst_path = os.path.join(dst_dir, file)
        copyfile(src_path, dst_path)


def create_data_descriptor_csv(data_dir, output_csv_path):
    """
    create CSV with data description
    :param data_dir: string, path to data folder
    :param output_csv_path: string, path to output folder
    """
    paths = get_files_recur(data_dir, suffix='jpg')
    categ = [p.split('/')[-2] for p in paths]
    data_desc_df = pd.DataFrame({'path': paths, 'category': categ})
    data_desc_df.to_csv(output_csv_path, index=False)
