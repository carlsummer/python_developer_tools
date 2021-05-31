import numpy
import os
import ctypes

#
# compile our function as a native shared library and load it in Python
#

src = '''float compute_mean(float* A, int n)
{
	int i;
	float sum = 0.0f;
	for(i=0; i<n; ++i)
		sum += A[i];
	return sum/n;
}'''

f = open('lib.c', 'w')
f.write(src)
f.close()
os.system('cc lib.c -fPIC -shared -o lib.so')
os.system('rm lib.c')
lib = ctypes.cdll.LoadLibrary('./lib.so')
os.system('rm lib.so')

# return value of `compute_mean` is a 32-bit float
lib.compute_mean.restype = ctypes.c_float

#
# generate a random array of length n
#

n = 1024
A = numpy.random.rand(n).astype(numpy.float32)

#
# compute the result
#
import time
start = time.time()
mean_ntv = lib.compute_mean(ctypes.c_void_p(A.ctypes.data), ctypes.c_int(n))
print(time.time()-start)

start = time.time()
mean_npy = numpy.mean(A)
print(time.time()-start)

print('* mean computed with native code: %.6f' % mean_ntv)
print('* mean computed with numpy  code: %.6f' % mean_npy)