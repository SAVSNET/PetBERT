import pandas as pd
import glob
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--binary_model_folder',
                    help='location of binary models', default="data/binary")
parser.add_argument('--save_dir', help="Where the merged dataset will be saved",
                    default="data/PetBERT_ICD/combined")
args = vars(parser.parse_args())


def create_dataset(dataset_type, label_lists):
    print(f'Importing {dataset_type} Dataset')
    df_base = pd.read_csv(f"{label_lists[0]}/multi_label_{dataset_type}.csv")

    for label_list in tqdm(label_lists[1:], desc=f"Creating {dataset_type} Dataset"):
        print("####### " + label_list + " #######")
        df_datalab = pd.read_csv(
            label_list + f"/multi_label_{dataset_type}.csv")
        Labels = f"{label_list}_Labels"
        df_datalab = df_datalab[['savsnet_consult_id', Labels]]
        df_base = df_base.merge(df_datalab.drop_duplicates(
            subset=['savsnet_consult_id']), on="savsnet_consult_id", how='left')

    return df_base


label_list = glob.glob(args+"/*")

df_train = create_dataset("train", label_list)
df_eval = create_dataset("eval", label_list)
df_test = create_dataset("test", label_list)

df_train.to_csv(args['save_dir'] + "train.csv")
df_eval.to_csv(args['save_dir'] + "eval.csv")
df_test.to_csv(args['save_dir'] + "test.csv")
