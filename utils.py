from datetime import *
import os
import pandas as pd


def get_time(date_string, date_format= "%Y-%m-%d"):
    """
    :param date_string:
    :param date_format:
    :return: formatted datetime
    """
    return datetime.strptime(date_string, date_format).date()


def load_data(dir_dataset, week_range):
    """

    :param dir_dataset: directory that contains one folder for each week. In each dir_dataset/week/ there should be a .csv file
                        for each GISAID sequence deposited in that week, containing the kmer profile of that sequence (see ./dataset_week directory for a sample)
    :param week_range: week range
    :return: load sequences in the week range
    """
    week_range = [str(x) for x in week_range]
    weeks_folder = [x for x in os.listdir(dir_dataset) if x in week_range]
    df_list = []
    w_list = []
    for week in weeks_folder:
        df_path = dir_dataset  + week +'/week_dataset.txt'
        df = pd.read_csv(df_path, header=None)
        # df = df[~df.iloc[:, 0].isin(id_unknown)]
        df_list.append(df)
        w_list += [week]*df.shape[0]
    return pd.concat(df_list), w_list

def map_variant_to_finalclass(class_list, non_neutral):
    # -1 -> non-neutral
    # 1 -> neutral
    final_class_list = []
    for c in class_list:
        if c in non_neutral:
            final_class_list.append(-1)
        else:
            final_class_list.append(1)
    return final_class_list

def get_variant_class(metadata, id_list):
    variant_name_list = []
    for id in id_list:
        variant_name_list.append(metadata[metadata['Accession.ID'] == id]['Variant'].values[0])
    return variant_name_list