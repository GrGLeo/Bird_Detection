import os
import shutil

def train_test_directory():
    # Get path to raw_data
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_path = os.path.join(path,"data\\raw_data\\training_set")
    dir_list = os.listdir(data_path)

    # Path to training and test set
    training_path = os.path.join(path,"data\\training")
    test_path = os.path.join(path,"data\\test")

    # Create new directory for training and test imgs
    if not os.path.exists(training_path):
        os.mkdir(training_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    for dirs in dir_list:
        dir_path = os.path.join(data_path,dirs)
        files_list = os.listdir(dir_path)
        num_files = len(files_list)
        num_train_files = int(0.9 * num_files)

        # Create classe specific directory
        train_classe_path = os.path.join(training_path, dirs)
        if not os.path.exists(train_classe_path):
            os.mkdir(train_classe_path)
        test_classe_path = os.path.join(test_path, dirs)
        if not os.path.exists(test_classe_path):
            os.mkdir(test_classe_path)
        

        # Move the file to the train directory
        for i, file_name in enumerate(files_list):
            if i < num_train_files:
                src_path = os.path.join(dir_path, file_name)
                dest_path = os.path.join(train_classe_path, file_name)
                shutil.move(src_path, dest_path)

        # Move the file to the test directory
        for i, file_name in enumerate(files_list):
            if i >= num_train_files:
                src_path = os.path.join(dir_path, file_name)
                dest_path = os.path.join(test_classe_path, file_name)
                shutil.move(src_path, dest_path)

if __name__ == "__main__":
    
    train_test_directory()
    print("All files moved")