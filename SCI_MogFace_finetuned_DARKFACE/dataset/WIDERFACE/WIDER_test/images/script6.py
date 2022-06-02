import os
for i in range(1,10):
    os.mkdir(f'./{i}')

files = []

for path, dir_list, file_list in os.walk('./0'):
    files += file_list

assert len(files) == 4000

for i in range(1,10):
    cf = files[i*400 : (i+1)*400]
    for filename in cf:
        os.rename(f'./0/{filename}', f'./{i}/{filename}')