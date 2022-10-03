#去除相关性

import numpy as np
from numpy import matrix as mat
from numpy import *;#导入numpy的库函数

H1,H2,H3,H4,H5,H6,H7,H8 = np.loadtxt('bin8_h_4000_Om0.3.txt', unpack=True, usecols=(0, 1,2,3,4,5,6,7))

x=np.vstack((H1,H2,H3,H4,H5,H6,H7,H8))
H_cov=np.cov(x)
print("H_cov\n",H_cov)
H_inv=np.linalg.inv(H_cov)

lam, matO = np.linalg.eig(H_inv)
_05lam=lam**0.5
data6=mat(diag(_05lam))
matO_T=np.transpose(matO)

data=np.dot(matO,data6)
transM=np.dot(data,matO_T)

sum=transM.sum(axis=1)
for i in range (len(sum)):
    transM[i]=transM[i]/sum[i]
print("transM\n",transM)

H_new=np.dot(transM,x)
H_new=H_new.T
print("H_new\n",H_new)
cov_new=np.cov(H_new.T)
print("cov_new\n",cov_new)
np.savetxt('bin8_h_decorr.txt', H_new, fmt='%5.6f')