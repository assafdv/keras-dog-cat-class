import os
from preproc.dog_cat_preproc import preproc, create_data_descriptor_csv
from train_test_pipeline.train_binary import train_dog_cat, preview_augmentation
from train_test_pipeline.test_binary import test

if __name__ == '__main__':
    wp = '/home/assaf/seetree/data/public/dogs_cats'

    # pre-process
    img_path = os.path.join(wp, 'train')
    train_dir = os.path.join(wp, 'subset_1k', 'train')
    val_dir = os.path.join(wp, 'subset_1k', 'validation')
    # preproc(wp, img_path)
    # create_data_descriptor_csv(train_dir, os.path.join(train_dir, 'train.csv'))
    # create_data_descriptor_csv(val_dir, os.path.join(val_dir, 'validation.csv'))

    # preview augmentation
    preview_dir = os.path.join(wp, 'subset_1k', 'preview')
    single_img_path = os.path.join(img_path, 'cat.0.jpg')
    # preview_augmentation(img_file_path=single_img_path, output_dir=preview_dir, nb_iters=20)

    # train
    # train_dog_cat(train_dir, val_dir)

    # evaluate results
    output_csv = os.path.join(wp, 'subset_1k', 'results', 'test_results.csv')
    labels_index = {"cats": 0, "dogs": 1}
    test(model_file='convnet_yann_Lecun1.h5',
         data_desc_csv=os.path.join(val_dir, 'validation.csv'),
         output_csv=output_csv,
         labels_index = labels_index)

