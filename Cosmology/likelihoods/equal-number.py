def lnlike(theta):
    sigma2_59 = mu_err_59**2
    
    x2_SN_59 =np.sum(np.power(mu_59 - mu_theory_bin7(redshift_59, theta), 2)/sigma2_59)
    
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
    
    x2_SN_1701=np.dot(np.dot(det_miu,C_1),det_miu.T)
    
    x2_SN=x2_SN_1701+x2_SN_59
    
    x2_Hz_bin3=Hubble_bin3(H_redshift_bin3,Hz_bin3,Hzerr_bin3,theta)
    x2_Hz_bin4=Hubble_bin4(H_redshift_bin4,Hz_bin4,Hzerr_bin4,theta)
    x2_Hz_bin5=Hubble_bin5(H_redshift_bin5,Hz_bin5,Hzerr_bin5,theta)
    x2_Hz_bin6=Hubble_bin6(H_redshift_bin6,Hz_bin6,Hzerr_bin6,theta)
    x2_Hz_bin7=Hubble_bin7(H_redshift_bin7,Hz_bin7,Hzerr_bin7,theta)
    
    x2_Hz_33=x2_Hz_bin3+x2_Hz_bin4+x2_Hz_bin5+x2_Hz_bin6+x2_Hz_bin7
    
    x2_Hz=x2_Hz_33
    
    x2=x2_SN+x2_Hz
                         
    lp =  -0.5*x2
    return lp