lines = []
for i in range(10):
    with open(f'./val_test{i}.txt', 'r') as f:
        lines += f.readlines()
with open(f'./val_test.txt', 'w') as f:
    f.writelines(lines)