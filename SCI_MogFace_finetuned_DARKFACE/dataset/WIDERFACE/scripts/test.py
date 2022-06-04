from base64 import encode
import os
import cv2
import hashlib

sets = []

for i in range(20):
    print(i)
    curset = set()
    for path, dir_list, file_list in os.walk(f'./{i%10}--Whatever'):
        for file_name in file_list:
            if 'png' in file_name:
                with open(os.path.join(path, file_name), 'rb') as f:
                    h = hashlib.sha256(f.read()).hexdigest()
                    # print(type(h), h)
                    curset.add(h)

    sets.append(curset)
    print(len(curset))

for i in range(10):
    assert len(sets[i]&sets[i+10]) == len(sets[i])

for i in range(10):
    for j in range(10):
        if i<j:
            assert len(sets[i] & sets[j]) == 0