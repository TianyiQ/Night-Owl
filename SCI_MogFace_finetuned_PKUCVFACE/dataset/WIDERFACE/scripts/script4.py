with open('./wider_val.txt', 'r') as oneline:
    onelines = oneline.readlines()

multilines = []
for line in onelines:
    eles = line.split(' ')
    assert eles[1][0] in '0123456789'
    multilines.append(eles[0].replace('/code/data/WIDER/WIDER_train/images/', '').replace('/code/data/WIDER/WIDER_val/images/', '') + '\n')
    multilines.append(str(int(eles[1])) + '\n')
    for i in range(int(eles[1])):
        a,b,c,d = int(eles[i*5+2]), int(eles[i*5+3]), int(eles[i*5+4]), int(eles[i*5+5])
        multilines.append(f'{a} {b} {c} {d} 0 0 0 0 0 0\n')

with open('./wider_face_val_bbx_gt.txt', 'w') as multiline:
    multiline.writelines(multilines)

# /code/data/WIDER/WIDER_train/images/1--Whatever/Whatever_3088.png 4 380 405 20 23 1 318 423 19 12 1 714 406 8 11 1 592 406 19 15 1