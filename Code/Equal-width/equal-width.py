import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import scipy.optimize as opt
from scipy.integrate import quad
import emcee
from multiprocessing import Pool
import pandas as pd
import time 
 
global Speed_of_Light
Speed_of_Light = 2.99792458e10 #cm/s


global Omega_m_true ,Hz
Omega_m_true = 0.3
h1_true=70
h2_true=70
h3_true=70
h4_true=70
h5_true=70
h6_true=70
h7_true=70
h8_true=70



#读取文件

global redshift_bin1,mu_bin1,mu_err_bin1
global redshift_bin2,mu_bin2,mu_err_bin2
global redshift_bin3,mu_bin3,mu_err_bin3
global redshift_bin4,mu_bin4,mu_err_bin4
global redshift_bin5,mu_bin5,mu_err_bin5
global redshift_bin6,mu_bin6,mu_err_bin6
global redshift_bin7,mu_bin7,mu_err_bin7
global redshift_bin8,mu_bin8,mu_err_bin8

redshift_bin1 ,mu_bin1, mu_err_bin1  = np.loadtxt('SNbin1.txt',unpack=True, usecols=(0,1,2))
redshift_bin2 ,mu_bin2, mu_err_bin2  = np.loadtxt('SNbin2.txt',unpack=True, usecols=(0,1,2))
redshift_bin3 ,mu_bin3, mu_err_bin3  = np.loadtxt('SNbin3.txt',unpack=True, usecols=(0,1,2))
redshift_bin4 ,mu_bin4, mu_err_bin4  = np.loadtxt('SNbin4.txt',unpack=True, usecols=(0,1,2))
redshift_bin5 ,mu_bin5, mu_err_bin5  = np.loadtxt('SNbin5.txt',unpack=True, usecols=(0,1,2))
redshift_bin6 ,mu_bin6, mu_err_bin6  = np.loadtxt('SNbin6.txt',unpack=True, usecols=(0,1,2))
redshift_bin7 ,mu_bin7, mu_err_bin7  = np.loadtxt('SNbin7.txt',unpack=True, usecols=(0,1,2))
redshift_bin8 ,mu_bin8, mu_err_bin8  = np.loadtxt('SNbin8.txt',unpack=True, usecols=(0,1,2))
redshift_59 ,mu_59, mu_err_59  = np.loadtxt('SN59.txt',unpack=True, usecols=(0,1,2))

global H_redshift_bin1,Hz_bin1,Hzerr_bin1
global H_redshift_bin2,Hz_bin2,Hzerr_bin2
global H_redshift_bin3,Hz_bin3,Hzerr_bin3
global H_redshift_bin4,Hz_bin4,Hzerr_bin4
global H_redshift_bin5,Hz_bin5,Hzerr_bin5
global H_redshift_bin6,Hz_bin6,Hzerr_bin6
global H_redshift_bin7,Hz_bin7,Hzerr_bin7
global H_redshift_bin8,Hz_bin8,Hzerr_bin8

H_redshift_bin1,Hz_bin1,Hzerr_bin1 = np.loadtxt('Hzbin1.txt', unpack=True, usecols=(0, 1,2))
H_redshift_bin2,Hz_bin2,Hzerr_bin2 = np.loadtxt('Hzbin2.txt', unpack=True, usecols=(0, 1,2))
H_redshift_bin3,Hz_bin3,Hzerr_bin3 = np.loadtxt('Hzbin3.txt', unpack=True, usecols=(0, 1,2))
H_redshift_bin4,Hz_bin4,Hzerr_bin4 = np.loadtxt('Hzbin4_new.txt', unpack=True, usecols=(0, 1,2))
H_redshift_bin5,Hz_bin5,Hzerr_bin5 = np.loadtxt('Hzbin5_new.txt', unpack=True, usecols=(0, 1,2))
H_redshift_bin6,Hz_bin6,Hzerr_bin6 = np.loadtxt('Hzbin6_new.txt', unpack=True, usecols=(0, 1,2))
H_redshift_bin7,Hz_bin7,Hzerr_bin7 = np.loadtxt('Hzbin7.txt', unpack=True, usecols=(0, 1,2))
H_redshift_bin8,Hz_bin8,Hzerr_bin8 = np.loadtxt('Hzbin8.txt', unpack=True, usecols=(0, 1,2))

c_sys=np.loadtxt('C_sys_new.txt')
C=c_sys
C_1=np.linalg.inv(C)

bins=[0,0.1,0.2,0.3,0.4,0.55,0.7,1.0,2.4]

def inv_E(z, theta ):  
    Omega_m = 0.3
    Omega_lambda= 1-Omega_m
    shang=3*(1+z)**2*Omega_m
    xia=np.sqrt((1+z)**3*Omega_m+Omega_lambda)
    return 0.5*shang/xia

def Integrate(z1,z2,theta):    
    integrate = quad(inv_E, z1, z2, args=(theta))[0]
    return integrate

#计算不同Bin的Hz
def binHz_bin1(z,theta):
    H1=theta[0]
    Hz=H1*Integrate(0,z,theta)+H1
    return Hz
def binHz_bin2(z,theta):
    H1=theta[0]
    H2=theta[1]
    I1=Integrate(0,bins[1],theta)
    Hz=H1*I1+H2*Integrate(bins[1],z,theta)+H2
    return Hz
def binHz_bin3(z,theta):
    H1=theta[0]
    H2=theta[1]
    H3=theta[2]
    I1=Integrate(0,bins[1],theta)
    I2=Integrate(bins[1],bins[2],theta)
    Hz=H1*I1+H2*I2+H3*Integrate(bins[2],z,theta)+H3
    return Hz
def binHz_bin4(z,theta):
    H1=theta[0]
    H2=theta[1]
    H3=theta[2]
    H4=theta[3]
    I1=Integrate(0,bins[1],theta)
    I2=Integrate(bins[1],bins[2],theta)
    I3=Integrate(bins[2],bins[3],theta)
    Hz=H1*I1+H2*I2+H3*I3+H4*Integrate(bins[3],z,theta)+H4
    return Hz
def binHz_bin5(z,theta):
    H1=theta[0]
    H2=theta[1]
    H3=theta[2]
    H4=theta[3]
    H5=theta[4]
    I1=Integrate(0,bins[1],theta)
    I2=Integrate(bins[1],bins[2],theta)
    I3=Integrate(bins[2],bins[3],theta)
    I4=Integrate(bins[3],bins[4],theta)
    Hz=H1*I1+H2*I2+H3*I3+H4*I4+H5*Integrate(bins[4],z,theta)+H5
    return Hz
def binHz_bin6(z,theta):
    H1=theta[0]
    H2=theta[1]
    H3=theta[2]
    H4=theta[3]
    H5=theta[4]
    H6=theta[5]
    I1=Integrate(0,bins[1],theta)
    I2=Integrate(bins[1],bins[2],theta)
    I3=Integrate(bins[2],bins[3],theta)
    I4=Integrate(bins[3],bins[4],theta)
    I5=Integrate(bins[4],bins[5],theta)
    Hz=H1*I1+H2*I2+H3*I3+H4*I4+H5*I5+H6*Integrate(bins[5],z,theta)+H6
    return Hz
def binHz_bin7(z,theta):
    H1=theta[0]
    H2=theta[1]
    H3=theta[2]
    H4=theta[3]
    H5=theta[4]
    H6=theta[5]
    H7=theta[6]
    I1=Integrate(0,bins[1],theta)
    I2=Integrate(bins[1],bins[2],theta)
    I3=Integrate(bins[2],bins[3],theta)
    I4=Integrate(bins[3],bins[4],theta)
    I5=Integrate(bins[4],bins[5],theta)
    I6=Integrate(bins[5],bins[6],theta)
    Hz=H1*I1+H2*I2+H3*I3+H4*I4+H5*I5+H6*I6+H7*Integrate(bins[6],z,theta)+H7
    return Hz
def binHz_bin8(z,theta):
    H1=theta[0]
    H2=theta[1]
    H3=theta[2]
    H4=theta[3]
    H5=theta[4]
    H6=theta[5]
    H7=theta[6]
    H8=theta[7]
    I1=Integrate(0,bins[1],theta)
    I2=Integrate(bins[1],bins[2],theta)
    I3=Integrate(bins[2],bins[3],theta)
    I4=Integrate(bins[3],bins[4],theta)
    I5=Integrate(bins[4],bins[5],theta)
    I6=Integrate(bins[5],bins[6],theta)
    I7=Integrate(bins[6],bins[7],theta)
    Hz=H1*I1+H2*I2+H3*I3+H4*I4+H5*I5+H6*I6+H7*I7+H8*Integrate(bins[7],z,theta)+H8
    return Hz

#被积函数1/H（z）
def daobinHz_bin1(z,theta):
    a=binHz_bin1(z,theta)
    return 1/a 
def daobinHz_bin2(z,theta):
    a=binHz_bin2(z,theta)
    return 1/a 
def daobinHz_bin3(z,theta):
    a=binHz_bin3(z,theta)
    return 1/a 
def daobinHz_bin4(z,theta):
    a=binHz_bin4(z,theta)
    return 1/a 
def daobinHz_bin5(z,theta):
    a=binHz_bin5(z,theta)
    return 1/a 
def daobinHz_bin6(z,theta):
    a=binHz_bin6(z,theta)
    return 1/a 
def daobinHz_bin7(z,theta):
    a=binHz_bin7(z,theta)
    return 1/a 
def daobinHz_bin8(z,theta):
    a=binHz_bin8(z,theta)
    return 1/a 

#积分
def Integrate_Hz_bin1(z,theta):
    integrate1 = np.zeros(len(z))
    result = np.zeros(len(z))
    for i in range(len(z)):
        integrate1[i]=quad(daobinHz_bin1 , 0.0 , z[i], args=(theta))[0]
        result[i]= integrate1[i]
    return result
def Integrate_Hz_bin2(z,theta):
    integrate2 = np.zeros(len(z))
    result = np.zeros(len(z))
    integrate1=quad(daobinHz_bin1 , 0.0 , bins[1], args=(theta))[0]
    for i in range(len(z)):
        integrate2[i]=quad(daobinHz_bin2 , bins[1] , z[i], args=(theta))[0]
        result[i]= integrate1+integrate2[i]
    return result
def Integrate_Hz_bin3(z,theta):
    integrate3 = np.zeros(len(z))
    result = np.zeros(len(z))
    integrate1=quad(daobinHz_bin1 , 0.0 , bins[1], args=(theta))[0]
    integrate2=quad(daobinHz_bin2 , bins[1] , bins[2], args=(theta))[0]
    for i in range(len(z)):
        integrate3[i]=quad(daobinHz_bin3 , bins[2] , z[i], args=(theta))[0]
        result[i]= integrate1+integrate2+integrate3[i]
    return result
def Integrate_Hz_bin4(z,theta):
    integrate4 = np.zeros(len(z))
    result = np.zeros(len(z))
    integrate1=quad(daobinHz_bin1 , 0.0 , bins[1], args=(theta))[0]
    integrate2=quad(daobinHz_bin2 , bins[1] , bins[2], args=(theta))[0]
    integrate3=quad(daobinHz_bin3 , bins[2] , bins[3], args=(theta))[0]
    for i in range(len(z)):
        integrate4[i]=quad(daobinHz_bin4 , bins[3] , z[i], args=(theta))[0]
        result[i]= integrate1+integrate2+integrate3+integrate4[i]
    return result
def Integrate_Hz_bin5(z,theta):
    integrate5 = np.zeros(len(z))
    result = np.zeros(len(z))
    integrate1=quad(daobinHz_bin1 , 0.0 , bins[1], args=(theta))[0]
    integrate2=quad(daobinHz_bin2 , bins[1] , bins[2], args=(theta))[0]
    integrate3=quad(daobinHz_bin3 , bins[2] , bins[3], args=(theta))[0]
    integrate4=quad(daobinHz_bin4 , bins[3] , bins[4], args=(theta))[0]    
    for i in range(len(z)):
        integrate5[i]=quad(daobinHz_bin5 , bins[4] , z[i], args=(theta))[0]
        result[i]= integrate1+integrate2+integrate3+integrate4+integrate5[i]
    return result
def Integrate_Hz_bin6(z,theta):
    integrate6 = np.zeros(len(z))
    result = np.zeros(len(z))
    integrate1=quad(daobinHz_bin1 , 0.0 , bins[1], args=(theta))[0]
    integrate2=quad(daobinHz_bin2 , bins[1] , bins[2], args=(theta))[0]
    integrate3=quad(daobinHz_bin3 , bins[2] , bins[3], args=(theta))[0]
    integrate4=quad(daobinHz_bin4 , bins[3] , bins[4], args=(theta))[0]
    integrate5=quad(daobinHz_bin5 , bins[4] , bins[5], args=(theta))[0]
    for i in range(len(z)):
        integrate6[i]=quad(daobinHz_bin6 , bins[5] , z[i], args=(theta))[0]
        result[i]= integrate1+integrate2+integrate3+integrate4+integrate5+integrate6[i]
    return result
def Integrate_Hz_bin7(z,theta):
    integrate7 = np.zeros(len(z))
    result = np.zeros(len(z))
    integrate1=quad(daobinHz_bin1 , 0.0 , bins[1], args=(theta))[0]
    integrate2=quad(daobinHz_bin2 , bins[1] , bins[2], args=(theta))[0]
    integrate3=quad(daobinHz_bin3 , bins[2] , bins[3], args=(theta))[0]
    integrate4=quad(daobinHz_bin4 , bins[3] , bins[4], args=(theta))[0]
    integrate5=quad(daobinHz_bin5 , bins[4] , bins[5], args=(theta))[0]
    integrate6=quad(daobinHz_bin6 , bins[5] , bins[6], args=(theta))[0]    
    for i in range(len(z)):
        integrate7[i]=quad(daobinHz_bin7 , bins[6] , z[i], args=(theta))[0]
        result[i]= integrate1+integrate2+integrate3+integrate4+integrate5+integrate6+integrate7[i]
    return result
def Integrate_Hz_bin8(z,theta):
    integrate8 = np.zeros(len(z))
    result = np.zeros(len(z))
    integrate1=quad(daobinHz_bin1 , 0.0 , bins[1], args=(theta))[0]
    integrate2=quad(daobinHz_bin2 , bins[1] , bins[2], args=(theta))[0]
    integrate3=quad(daobinHz_bin3 , bins[2] , bins[3], args=(theta))[0]
    integrate4=quad(daobinHz_bin4 , bins[3] , bins[4], args=(theta))[0]
    integrate5=quad(daobinHz_bin5 , bins[4] , bins[5], args=(theta))[0]
    integrate6=quad(daobinHz_bin6 , bins[5] , bins[6], args=(theta))[0]
    integrate7=quad(daobinHz_bin7 , bins[6] , bins[7], args=(theta))[0]    
    for i in range(len(z)):
        integrate8[i]=quad(daobinHz_bin8 , bins[7] , z[i], args=(theta))[0]
        result[i]= integrate1+integrate2+integrate3+integrate4+integrate5+integrate6+integrate7+integrate8[i]
    return result

#光度距离
def luminosity_distance_bin1(z, theta):
    a=(Speed_of_Light/1e5)*(1+z)
    dl=a*Integrate_Hz_bin1(z,theta)
    return dl
def luminosity_distance_bin2(z, theta):
    a=(Speed_of_Light/1e5)*(1+z)
    dl=a*Integrate_Hz_bin2(z,theta)
    return dl
def luminosity_distance_bin3(z, theta):
    a=(Speed_of_Light/1e5)*(1+z)
    dl=a*Integrate_Hz_bin3(z,theta)
    return dl
def luminosity_distance_bin4(z, theta):
    a=(Speed_of_Light/1e5)*(1+z)
    dl=a*Integrate_Hz_bin4(z,theta)
    return dl
def luminosity_distance_bin5(z, theta):
    a=(Speed_of_Light/1e5)*(1+z)
    dl=a*Integrate_Hz_bin5(z,theta)
    return dl
def luminosity_distance_bin6(z, theta):
    a=(Speed_of_Light/1e5)*(1+z)
    dl=a*Integrate_Hz_bin6(z,theta)
    return dl
def luminosity_distance_bin7(z, theta):
    a=(Speed_of_Light/1e5)*(1+z)
    dl=a*Integrate_Hz_bin7(z,theta)
    return dl
def luminosity_distance_bin8(z, theta):
    a=(Speed_of_Light/1e5)*(1+z)
    dl=a*Integrate_Hz_bin8(z,theta)
    return dl

def lnprior(theta):
    H1=theta[0]
    H2=theta[1]
    H3=theta[2]
    H4=theta[3]
    H5=theta[4]
    H6=theta[5]
    H7=theta[6]
    H8=theta[7]
    if  50<H1<80 and 50<H2<80 and 50<H3<80 and 50<H4<80 and 50<H5<80 and 50<H6<80 and 50<H7<80 and 50<H8<80 :
        return 0.0
    return -np.inf

#理论距离模量
def mu_theory_bin1(z, theta):
    return 5*np.log10(luminosity_distance_bin1(z, theta)) + 25
def mu_theory_bin2(z, theta):
    return 5*np.log10(luminosity_distance_bin2(z, theta)) + 25
def mu_theory_bin3(z, theta):
    return 5*np.log10(luminosity_distance_bin3(z, theta)) + 25
def mu_theory_bin4(z, theta):
    return 5*np.log10(luminosity_distance_bin4(z, theta)) + 25
def mu_theory_bin5(z, theta):
    return 5*np.log10(luminosity_distance_bin5(z, theta)) + 25
def mu_theory_bin6(z, theta):
    return 5*np.log10(luminosity_distance_bin6(z, theta)) + 25
def mu_theory_bin7(z, theta):
    return 5*np.log10(luminosity_distance_bin7(z, theta)) + 25
def mu_theory_bin8(z, theta):
    return 5*np.log10(luminosity_distance_bin8(z, theta)) + 25

#计算Hz的卡方
def Hubble_bin1(H_redshift,Hz,Hzerr,theta):
    chi=0
    for i in range(len(Hz)):
        chi+=np.power((binHz_bin1(H_redshift[i],theta)-Hz[i]),2)/(Hzerr[i]*Hzerr[i])
    return chi
def Hubble_bin2(H_redshift,Hz,Hzerr,theta):
    chi=0
    for i in range(len(Hz)):
        chi+=np.power((binHz_bin2(H_redshift[i],theta)-Hz[i]),2)/(Hzerr[i]*Hzerr[i])
    return chi
def Hubble_bin3(H_redshift,Hz,Hzerr,theta):
    chi=0
    for i in range(len(Hz)):
        chi+=np.power((binHz_bin3(H_redshift[i],theta)-Hz[i]),2)/(Hzerr[i]*Hzerr[i])
    return chi
def Hubble_bin4(H_redshift,Hz,Hzerr,theta):
    chi=0
    for i in range(len(Hz)):
        chi+=np.power((binHz_bin4(H_redshift[i],theta)-Hz[i]),2)/(Hzerr[i]*Hzerr[i])
    return chi
def Hubble_bin5(H_redshift,Hz,Hzerr,theta):
    chi=0
    for i in range(len(Hz)):
        chi+=np.power((binHz_bin5(H_redshift[i],theta)-Hz[i]),2)/(Hzerr[i]*Hzerr[i])
    return chi
def Hubble_bin6(H_redshift,Hz,Hzerr,theta):
    chi=0
    for i in range(len(Hz)):
        chi+=np.power((binHz_bin6(H_redshift[i],theta)-Hz[i]),2)/(Hzerr[i]*Hzerr[i])
    return chi
def Hubble_bin7(H_redshift,Hz,Hzerr,theta):
    chi=0
    for i in range(len(Hz)):
        chi+=np.power((binHz_bin7(H_redshift[i],theta)-Hz[i]),2)/(Hzerr[i]*Hzerr[i])
    return chi
def Hubble_bin8(H_redshift,Hz,Hzerr,theta):
    chi=0
    for i in range(len(Hz)):
        chi+=np.power((binHz_bin8(H_redshift[i],theta)-Hz[i]),2)/(Hzerr[i]*Hzerr[i])
    return chi

def lnlike(theta):
    sigma2_59 = mu_err_59**2
    x2_SN_59 =np.sum(np.power((mu_59 - mu_theory_bin7(redshift_59, theta)), 2)/sigma2_59)
    
    det_miu_bin1=mu_bin1 - mu_theory_bin1(redshift_bin1, theta)
    det_miu_bin2=mu_bin2 - mu_theory_bin2(redshift_bin2, theta)
    det_miu=np.append(det_miu_bin1,det_miu_bin2)
    det_miu_bin3=mu_bin3 - mu_theory_bin3(redshift_bin3, theta)
    det_miu=np.append(det_miu,det_miu_bin3)
    det_miu_bin4=mu_bin4 - mu_theory_bin4(redshift_bin4, theta)
    det_miu=np.append(det_miu,det_miu_bin4)
    det_miu_bin5=mu_bin5 - mu_theory_bin5(redshift_bin5, theta)
    det_miu=np.append(det_miu,det_miu_bin5)
    det_miu_bin6=mu_bin6 - mu_theory_bin6(redshift_bin6, theta)
    det_miu=np.append(det_miu,det_miu_bin6)
    det_miu_bin7=mu_bin7 - mu_theory_bin7(redshift_bin7, theta)
    det_miu=np.append(det_miu,det_miu_bin7)
    det_miu_bin8=mu_bin8 - mu_theory_bin8(redshift_bin8, theta)
    det_miu=np.append(det_miu,det_miu_bin8)
    x2_SN_1701=np.dot(np.dot(det_miu,C_1),det_miu.T)
    
    x2_SN=x2_SN_1701+x2_SN_59
    
    x2_Hz_bin1=Hubble_bin1(H_redshift_bin1,Hz_bin1,Hzerr_bin1,theta)
    x2_Hz_bin2=Hubble_bin2(H_redshift_bin2,Hz_bin2,Hzerr_bin2,theta)
    x2_Hz_bin3=Hubble_bin3(H_redshift_bin3,Hz_bin3,Hzerr_bin3,theta)
    x2_Hz_bin4=Hubble_bin4(H_redshift_bin4,Hz_bin4,Hzerr_bin4,theta)
    x2_Hz_bin5=Hubble_bin5(H_redshift_bin5,Hz_bin5,Hzerr_bin5,theta)
    x2_Hz_bin6=Hubble_bin6(H_redshift_bin6,Hz_bin6,Hzerr_bin6,theta)
    x2_Hz_bin7=Hubble_bin7(H_redshift_bin7,Hz_bin7,Hzerr_bin7,theta)
    x2_Hz_bin8=Hubble_bin8(H_redshift_bin8,Hz_bin8,Hzerr_bin8,theta)
    
    x2_Hz_36=x2_Hz_bin1+x2_Hz_bin2+x2_Hz_bin3+x2_Hz_bin4+x2_Hz_bin5+x2_Hz_bin6+x2_Hz_bin7+x2_Hz_bin8
    
    
    x2_Hz=x2_Hz_36
    
    x2=x2_SN+x2_Hz
                         
    lp =  -0.5*x2
    return lp


def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)    


if __name__ == "__main__":
    with Pool() as pool:
        canshu=[h1_true,h2_true,h3_true,h4_true,h5_true,h6_true,h7_true,h8_true]
        ndim, nwalkers = 8, 32
        p0 = [canshu + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(), pool = pool)
        print("Running MCMC...")
        t1 = time.time()
        sampler.run_mcmc(p0, 4000, progress = True)
        t2 = time.time()
        print("Done.",'The cost time is {}'.format(t2 - t1))
        
#save all the data
test_sampler = sampler.chain[:,500:,:].reshape(sampler.chain.shape[0]*(sampler.chain.shape[1]-500),8)
np.savetxt('bin8_h_4000_Om0.3.txt', test_sampler, fmt='%5.6f')
