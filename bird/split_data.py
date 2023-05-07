import os
import numpy as np
import shutil

def train_test_directory():
    # Get path to raw_data
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_path = os.path.join(path,"data\\raw_data\\training_set")
    dir_list = os.listdir(data_path)

    # Path to training, val and test set
    training_path = os.path.join(path,"data\\train")
    val_path = os.path.join(path,"data\\val")
    test_path = os.path.join(path,"data\\test")

    # Create new directory for training, val and test imgs
    if not os.path.exists(training_path):
        os.mkdir(training_path)
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    # Loop through directory
    num_original_files = []
    num_moved_files = []
    for dirs in dir_list:
        dir_path = os.path.join(data_path,dirs)
        files_list = os.listdir(dir_path)
        num_files = len(files_list)
        num_original_files.append(num_files)
        num_train_files = int(0.7 * num_files)
        rest = len(files_list) - num_train_files
        num_val_files = int(rest * (2/3) + num_train_files)


        # Create classe specific directory
        train_classe_path = os.path.join(training_path, dirs)
        if not os.path.exists(train_classe_path):
            os.mkdir(train_classe_path)
        val_classe_path = os.path.join(val_path, dirs)
        if not os.path.exists(val_classe_path):
            os.mkdir(val_classe_path)
        test_classe_path = os.path.join(test_path, dirs)
        if not os.path.exists(test_classe_path):
            os.mkdir(test_classe_path)
        

        # Move the file to the train directory
        for i, file_name in enumerate(files_list):
            if i < num_train_files:
                src_path = os.path.join(dir_path, file_name)
                dest_path = os.path.join(train_classe_path, file_name)
                shutil.move(src_path, dest_path)
            elif i >= num_train_files and i < num_val_files:
                src_path = os.path.join(dir_path, file_name)
                dest_path = os.path.join(val_classe_path, file_name)
                shutil.move(src_path, dest_path)
            elif i >= num_val_files:
                src_path = os.path.join(dir_path, file_name)
                dest_path = os.path.join(test_classe_path, file_name)
                shutil.move(src_path, dest_path)

        # Summing total files moved
        num_train_files = len(os.listdir(train_classe_path))
        num_val_files = len(os.listdir(val_classe_path))
        num_test_files = len(os.listdir(test_classe_path))
        num_moved_files.append(num_train_files+num_val_files+num_test_files)

    assert num_original_files == num_moved_files,"Files missing"

if __name__ == "__main__":
    
    train_test_directory()  
    print("All files moved")