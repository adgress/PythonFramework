from timer import timer
import sys
timer.tic()
x = int(sys.argv[2])
for i in range(int(sys.argv[2]),int(sys.argv[3])):
    x += 1
#print x
s = 'count_test_output_' + str(sys.argv[1])
#print s
#timer.toc()
#print 'sys.argv:' + str(sys.argv)
