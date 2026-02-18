# make dataset script

# -*- coding: utf-8 -*-
from operator import index
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import ast
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.transform import resize
from PIL import ImageFile
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True

@click.command() # NEW VERSION, had to change a bit, for it to be able to run, as we have a different directory structure
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('raw_images_dir', type=click.Path(exists=True)) # having to add this due to directories
@click.argument('output_dir', type=click.Path())
@click.argument('classes', default=None, type=str)
def main(input_filepath, raw_images_dir, output_dir, classes):
    """ Runs data processing scripts to turn raw data from (data/raw) into
        cleaned data ready to be analyzed (saved in data/processed).
    """
    print(f"!!!!!!!!!!!!!!!!!!!!!!!{os.getcwd()}")
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    os.makedirs(f"./{output_dir}/images", exist_ok=True)  
    assert os.path.isdir(f"./{output_dir}/images")

    logger.info(f'Processing labels and store the result in ./{output_dir}/processed_labels.csv')
    filter_and_process_labels(input_filepath, output_dir, classes)   
    assert os.path.exists(f"{output_dir}/processed_labels.csv")

    if os.listdir(f"./{output_dir}/images") == []:
        logger.info(f'Creating image dataset in ./{output_dir}/images')
        create_images(raw_images_dir, output_dir)         
        assert os.listdir(f"./{output_dir}/images")
    else:
        logger.info(f'./{output_dir}/images not empty, skipping creation of images')

    logger.info('Dataset is ready to be used!')


def filter_and_process_labels(input_filepath,output_filepath, df):
    base_df = pd.read_csv(f'{input_filepath}/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv',index_col=0)
    # invalid_images = pd.read_csv(f'{input_filepath}/Invalid_images.csv', header=None, index_col=0)
    print(f"!!!!!!!!!!!!!!!!!!!!!!!shape of the base_df: {base_df.shape}")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!the nans in df_no_nan: {base_df['Labels'].isnull().sum()}")

    # Excluding NaNs in the labels and in the patientbirth so we can make an age column
    df_no_nan = base_df[(~base_df["Labels"].isna()) & (~base_df["PatientBirth"].isna())]
    print(f"!!!!!!!!!!!!!!!!!!!!!!!the nans in df_no_nan: {df_no_nan['Labels'].isnull().sum()}")
    # Excluding labels including the 'suboptimal study' label
    df_no_clear_label = df_no_nan[~df_no_nan["Labels"].str.contains('suboptimal study')]
    df_no_clear_label = df_no_clear_label[~df_no_clear_label["Labels"].str.contains('exclude')]
    df_no_clear_label = df_no_clear_label[~df_no_clear_label["Labels"].str.contains('Unchanged')]
    
    # Keeping only the PA, AP and AP_horizontal projections
    df_view = df_no_clear_label[(df_no_clear_label['Projection'] == 'PA') | 
                                (df_no_clear_label['Projection'] == 'AP') | 
                                (df_no_clear_label['Projection'] == 'AP_horizontal')]

    df_view["PatientBirth"] = pd.to_datetime(df_view["PatientBirth"], format="%Y").dt.year
    df_view["StudyDate_DICOM"] = pd.to_datetime(df_view["StudyDate_DICOM"], format="%Y%m%d").dt.strftime("%d-%m-%Y")
    df_view["study_year"] = pd.to_datetime(df_view["StudyDate_DICOM"], format="%d-%m-%Y").dt.year
    df_view["age"] = df_view["study_year"] - df_view["PatientBirth"]

    # Stripping and lowercasing all individual labels
    stripped_lowercased_labels = []
    for label_list in list(df_view['Labels']):
        label_list = ast.literal_eval(label_list)
        prepped_labels = []
        for label in label_list:
            if label != '':
                new_label = label.strip(' ').lower()   # Stripping and lowercasing
                prepped_labels.append(new_label)
        # Removing label duplicates in this appending
        stripped_lowercased_labels.append(list(set(prepped_labels)))
    #print(df_view.head())

    # # Applying it to the preprocessed dataframe
    df_view['Labels'] = stripped_lowercased_labels
    #print(df_view.head())

    binary = []
    for labels in df_view["Labels"]:
        if "cardiomegaly" in labels:
            binary.append(1)
        else:
            binary.append(0)
    df_view["target_label"] = binary
    print(f"!!!!!!!!!!!!!!!!!!!!!!!shape of the base_df after processing: {df_view.shape}")


    # --- BUILD INVALID_IMAGES.CSV (full excluded rows, all columns) ---
    invalid_path = f'{input_filepath}/Invalid_images.csv'
    kept_ids = set(df_view['ImageID'].astype(str))
    excluded_rows = base_df[~base_df['ImageID'].astype(str).isin(kept_ids)].copy()
    excluded_rows.to_csv(invalid_path, index=False)   # <-- all columns retained

    # (Optional) also read it back if you still want to exclude using that file:
    invalid_images = pd.read_csv(invalid_path)
    df_no_invalid = df_view[~df_view['ImageID'].astype(str).isin(invalid_images['ImageID'].astype(str))].reset_index(drop=True)
    # In practice df_view is already the kept set, so:
    df_no_invalid = df_view.reset_index(drop=True)

    # --- SAVE KEPT ROWS ---
    os.makedirs(output_filepath, exist_ok=True)
    df_no_invalid.to_csv(f"{output_filepath}/processed_labels.csv", index=False)
    return df_no_invalid


# tqdm works, even though it says 3004 images
def create_images(input_filepath,output_filepath):
    #Load labels
    labels = pd.read_csv(f'{output_filepath}/processed_labels.csv')
    labels["Labels"] = labels["Labels"].apply(lambda x: ast.literal_eval(x))
    #Filter images to remove lateral views
    os.makedirs(f"./{output_filepath}/images", exist_ok=True)

    #Get images present at input_filepath
    images_path = glob.glob(f"./{input_filepath}/**/*.png",recursive=True)
    image_names = [path.split('/')[-1] for path in images_path]
    for idx,i_name in enumerate(tqdm(image_names, dynamic_ncols=True)):
        #Resize the image and save it in the processed folder
        if i_name in labels["ImageID"].unique():
            try:
                img = np.expand_dims(io.imread(images_path[idx]),-1)
                max_value = np.max(img) 
                img = tf.image.resize_with_pad(img, 512, 512)
                img = img/max_value
                tf.keras.utils.save_img(f"./{output_filepath}/images/{i_name}", img, scale=True, data_format="channels_last")  
            except Exception as e:
                print("corrupt image, couldn't be opened, therefore skip")
                continue

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

# RUN THIS: python make_dataset.py ../../data/external ../../data/raw ../../data/processed/padchest ""