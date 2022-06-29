from os import listdir
from os.path import isfile, join
mypath = './DSFD_zdce-V1/txts/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

with open('scarecrow.txt','w') as outfile:
    for infilename in onlyfiles:
        with open(join(mypath, infilename),'r') as infile:
            curline = '/dataset/image/' + infilename.replace('txt','png')
            inlines = infile.readlines()
            cnt = 0
            liststr = ''
            for line in inlines:
                if line[0] not in '0123456789.':
                    continue
                strs = line.split()
                if(float(strs[4]) > 0.5):
                    x1 = int(float(strs[0])+0.5)
                    y1 = int(float(strs[1])+0.5)
                    x2 = int(float(strs[2])+0.5)
                    y2 = int(float(strs[3])+0.5)
                    liststr += f' {x1} {y1} {x2-x1+1} {y2-y1+1} 1'
                    cnt += 1
            curline += f' {cnt}' + liststr
            outfile.write(curline + '\n')

