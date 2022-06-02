import os

with open('./val_train.txt', 'w') as f:
    for path, dir_list, file_list in os.walk('./WIDER_test/images'):
        for file_name in file_list:
            f.write(f'/code/dataset/WIDERFACE/WIDER_test/images/0/{file_name}\n')