import glob
import os
import pandas as pd

if __name__ == "__main__":

    jpg_files = glob.glob('./data/train/mutombo_cropped_aligned_margin_40/*/*.jpg')
    jpg_files_processed = [(os.path.join(*item.split(os.sep)[3:]), item.split(os.sep)[4]) for item in
                           jpg_files]

    jpg_files_processed_paths = [item[0] for item in jpg_files_processed]
    jpg_files_processed_id = [item[1] for item in jpg_files_processed]

    df_files = pd.DataFrame({'filePath': jpg_files_processed_paths, 'id': jpg_files_processed_id})

    mutombo_train_file = os.path.join('./data/train/mutombo_train_file.txt')
    print('Number of files after w/o thresholding = {}'.format(len(df_files)))
    df_files[["filePath", "id"]].to_csv(mutombo_train_file, sep=" ", header=False, index=False)

    df_files_grouped = df_files.groupby(["id"])["filePath"].count().to_frame("numPhotos")

    photo_thresholds = [20, 30, 50]

    for thresh in photo_thresholds:
        df_files_grouped_thresholded = df_files_grouped[df_files_grouped["numPhotos"] >= thresh]
        df_files_thresholded = df_files.merge(df_files_grouped_thresholded, left_on="id", right_index=True, how="inner")
        mutombo_train_file = os.path.join('./data/train/mutombo_train_file_{}.txt'.format(thresh))
        print('Number of files after thresholding with {} = {}'.format(thresh, len(df_files_thresholded)))
        df_files_thresholded[["filePath", "id"]].to_csv(mutombo_train_file, sep=" ", header=False, index=False)



