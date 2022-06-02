import os
import numpy as np
np.random.seed(123)

imgs=[]
with open('_all_.txt','r') as f:
    for s in f:
        if 'Whatever' in s:
            imgs.append([s.strip().split('/')[1]])
        elif s.strip().count(' ')>4:
            imgs[-1].append(s)
            
n=len(imgs)
np.random.shuffle(imgs)
for i in range(10):
    os.mkdir(f'./{i}--Whatever')
    for img in imgs[n*i//10:n*(i+1)//10]:
        os.system(f'mv __all__/{img[0]} ./{i}--Whatever/')