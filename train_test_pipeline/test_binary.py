import os
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import accuracy_score, confusion_matrix
from train_test_pipeline.constants import IM_HEIGHT, IM_WIDTH
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict(model, img_path, labels_index):
    """
    predict probability and class label (category) for a given image.
    :param model: Keras Model object
    :param img_path: string, path to image file.
    :return: dict with predictions.
    """
    # init
    index_labels = {d: k for k, d in labels_index.items()}

    # load image
    img = load_img(img_path, target_size=(IM_WIDTH, IM_HEIGHT))

    # pre-processing
    img_tensor = img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)   # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.

    # predict
    predict_prob = float(model.predict(img_tensor))
    predict_label = int(predict_prob > 0.5)
    predict_category = index_labels[predict_label]
    return {"path": img_path, "predict_prob": predict_prob, "predict_categ": predict_category}


def test(model_file, data_desc_csv, output_csv, labels_index):
    """
    Predict on test data and save results to CSV
    :param model_file: keras Model Object
    :param data_desc_csv: path to CSV file with data description (path, category)
    :param output_csv: path to output CSV file with predictions
    """
    # init
    df_results = pd.DataFrame(columns=['path', 'category', 'predict_categ', 'predict_prob'])

    # load model
    model = load_model(os.path.join('models', model_file))

    # load data desc
    data_desc_df = pd.read_csv(data_desc_csv)

    # get predictions
    for index, row in data_desc_df.iterrows():
        res_dict = predict(model, row['path'], labels_index)
        res_dict.setdefault('category', row['category'])
        df_results = df_results.append(res_dict, ignore_index=True)

    # statistics
    y_pred = df_results["predict_categ"]
    y_true = df_results["category"]
    logger.info('Confusion Matrix: {}'.format(confusion_matrix(y_true, y_pred)))
    logger.info('Accuracy: {}'.format(accuracy_score(y_true, y_pred)))

    # save output
    df_results.to_csv(output_csv, index=False)