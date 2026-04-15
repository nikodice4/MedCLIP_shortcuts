# make dataset script

# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import ast
import os

@click.command() # NEW VERSION, had to change a bit, for it to be able to run, as we have a different directory structure
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.argument('classes', default=None, type=str)
def main(input_filepath, output_dir, classes):
    """ Runs data processing scripts to turn raw data from (data/raw) into
        cleaned data ready to be analyzed (saved in data/processed).
    """
    print(f"!!!!!!!!!!!!!!!!!!!!!!!{os.getcwd()}")
    logger = logging.getLogger(__name__)
    logger.info('making padchest pneumothorax data set from raw data')

    logger.info(f'Processing labels and store the result in ./{output_dir}/processed_labels_px.csv')
    filter_and_process_labels(input_filepath, output_dir, classes)   
    assert os.path.exists(f"{output_dir}/processed_labels_px.csv")

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
        # if "cardiomegaly" in labels: # UNCOMMENT IF YOU WANT CARDIOMEGALY
        if "pneumothorax" in labels:
            binary.append(1)
        else:
            binary.append(0)
    df_view["target_label"] = binary
    print(f"!!!!!!!!!!!!!!!!!!!!!!!shape of the base_df after processing: {df_view.shape}")

    # --- SAVE KEPT ROWS ---
    os.makedirs(output_filepath, exist_ok=True)
    df_view.reset_index(drop=True).to_csv(f"{output_filepath}/processed_labels_px.csv", index=False)
    return df_view

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
