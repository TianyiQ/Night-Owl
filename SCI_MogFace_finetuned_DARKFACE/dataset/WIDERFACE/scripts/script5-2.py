import os

for i in range(10):
    with open(f'./val_test{i}.txt', 'w') as f:
        for path, dir_list, file_list in os.walk(f'./WIDER_test/images/{i}'):
            for file_name in file_list:
                f.write(f'/code/dataset/WIDERFACE/WIDER_test/images/{i}/{file_name}\n')