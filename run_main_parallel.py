from subprocess import call, Popen
from timer.timer import tic, toc
import time
tic()
num_to_count = 50000000
n = 4
p = [0]*n
split_size = num_to_count / n
for i in range(0,n):
    p[i] = Popen(['python',
                  'count_test.py',
                  str(i),
                  str(i*split_size),
                  str((i+1)*split_size)
                  ])
    #time.sleep(1)
for i in range(0,n):
    stdoutdata, stderrdata = p[i].communicate()
    #print stdoutdata
print 'All Done!'
toc()