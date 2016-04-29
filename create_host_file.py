import sys
s = 'hostfile'
num_nodes = int(sys.argv[1])
with open(s,'w') as f:
    f.write('master\n')
    for i in range(num_nodes):
        n = str(i+1)
        f.write('node' + n.zfill(3) + '\n')
