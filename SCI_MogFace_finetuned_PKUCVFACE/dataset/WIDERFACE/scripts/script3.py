with open('./wider_val.txt', 'r') as widerval:
    lines = widerval.readlines()
    with open('./val.txt', 'w') as val:
        val.writelines(line.split(' ')[0].replace('/code/data/WIDER', '/code/dataset/WIDERFACE') + '\n' for line in lines)