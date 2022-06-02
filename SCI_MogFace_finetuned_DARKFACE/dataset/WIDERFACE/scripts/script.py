import os
out = ''
for i in range(1,6001):
    with open(f'./{i}.txt','r') as infile:
        lines = infile.readlines()
        out += f'0--Whatever/0_Whatever_{i}.png\n' + lines[0]
        for i in range(1,len(lines)):
            nums = list(map(int,lines[i].split()[:4]))
            lines[i] = f'{nums[0]} {nums[1]} {nums[2]-nums[0]+1} {nums[3]-nums[1]+1} 0 0 0 0 0 0\n'
            out += lines[i]
with open(f'./wider_face_train_bbx_gt.txt','w') as outfile:
    outfile.write(out)