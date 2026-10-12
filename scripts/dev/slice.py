import sys
p, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ls = open(p, encoding='utf-8').readlines()
for i, l in enumerate(ls[a-1:b], start=a):
    print(f"{i}| {l.rstrip()}")
