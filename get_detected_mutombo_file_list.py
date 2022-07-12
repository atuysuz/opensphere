import glob
import os

if __name__ == "__main__":

    jpg_files = glob.glob('./data/train/mutombo_cropped_aligned_margin_40/*/*.jpg')
    jpg_files_processed = [os.path.join(*item.split(os.sep)[3:]) + " " + item.split(os.sep)[4] + "\n" for item in jpg_files]

    mutombo_train_file = './data/train/mutombo_train_file.txt'
    with open(mutombo_train_file, 'w') as fp:
        fp.writelines(jpg_files_processed)