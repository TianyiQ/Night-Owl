import os
import numpy as np
np.random.seed(123)

imgs=[]
with open('_all_.txt','r') as f:
    for s in f:
        s=s.strip()
        if 'Whatever' in s:
            imgs.append([s.split('/')[1]])
        elif s.count(' ')>4:
            imgs[-1].append(s)
            
n=len(imgs)
np.random.shuffle(imgs)
for i in range(10):
    os.mkdir(f'./{i}--Whatever')
    for t in range(n*i//10,n*(i+1)//10):
        name=imgs[t][0][2:]
        os.system(f'mv __all__/{imgs[t][0]} ./{i}--Whatever/{name}')
        imgs[t][0]=f'{i}--Whatever/{name}'
        
for i in range(10):
    train=imgs[:n*i//10]+imgs[n*(i+1)//10:]
    val=imgs[n*i//10:n*(i+1)//10]
    with open(f'./{i}--Whatever/train.txt','w') as f:
        for img in train:
            f.write(img[0]+'\n')
            f.write(str(len(img)-1)+'\n')
            for s in img[1:]:
                f.write(s+'\n')
    
    with open(f'./{i}--Whatever/val.txt','w') as f:
        for img in val:
            f.write(img[0]+'\n')
            f.write(str(len(img)-1)+'\n')
            for s in img[1:]:
                f.write(s+'\n')
    