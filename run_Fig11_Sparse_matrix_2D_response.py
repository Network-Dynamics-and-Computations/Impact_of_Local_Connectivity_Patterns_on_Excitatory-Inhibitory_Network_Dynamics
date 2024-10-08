# %%
import numpy as np
from numpy import linalg as la
import matplotlib.pylab as plt
from scipy.optimize import fsolve
import math
import statsmodels.api as sm
from scipy.stats import norm
import seaborn as sns
from scipy import stats
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from Meanfield_Dyn_util import *
from utils import *

# %%
import multiprocessing as mp

# %%
import scipy
from functools import partial
def odeIntegral(x,t,J,I=0):
    x = np.squeeze(x)
    x = np.reshape(x,(len(x),1))
    # dxdt = -x+J@np.tanh(x)#+I[0]
    dxdt = -x+J@x+I[0]
    return np.squeeze(dxdt)
def odesimulation(t,xinit,Jpt,I):
	return scipy.integrate.odeint(partial(odeIntegral,J=Jpt,I=I),xinit,t)
shiftx = 1.5
def odeIntegralP(x,t,J,I=0):
	x = np.squeeze(x)
	x = np.reshape(x,(len(x),1))
	# print('size:',np.shape(x),np.shape(J@np.tanh(x)))
	dxdt = -x+J@(1.0+np.tanh(x-shiftx))
	return np.squeeze(dxdt)
def odesimulationP(t,xinit,Jpt,I):
	return scipy.integrate.odeint(partial(odeIntegralP,J=Jpt,I=I),xinit,t)

# %%
### define the network parameters of the adjacency matrix
N  = 1500
Kt = int(N*0.2)
J = 1/np.sqrt(N)*0.5  ### TODO: make sure this scalar with David&Stefano's paper
J = 0.00325
ntau   = 10
trials = 3
tau_series = np.linspace(0,0.15,ntau)#np.linspace(0.1,0.2,ntau)#


g, gamma =  6, 1/4.0
NE = int(N/(1+gamma))
NI = N-NE#NE*gamma
N  = NE+NI ### update 
ALPHAE, ALPHAI = NE/N, NI/N
KE, KI = int(Kt/(1+gamma)), int(Kt/(1+gamma)*gamma) ### fixed out-degree
ce, ci = KE/NE, KI/NI
print('ce and ci:',ce,ci)
### assert that the differences between ce and ci are smaller than epsilon
epsilon = 1E-2
assert np.abs(ce-ci)<epsilon
# assert ce==ci
c = ce
ji,je = g*J,J 
### define the network parameters of the diluted Gaussian matrix 
ge, gi = np.sqrt(je**2*c*(1-c)*N), np.sqrt(ji**2*c*(1-c)*N) 
hat_sigmae, hat_sigmai = np.sqrt(c*(1-c)), np.sqrt(c*(1-c))### standard deviation of the adjacency matrix
sigmae,sigmai = np.sqrt(c*(1-c)*J**2*N), np.sqrt(c*(1-c)*(-g*J)**2*N)### with magnitude of the coupling
JE,JI = je*c*NE, ji*c*NI 
lambda0 = JE-JI 

    
#### constant and deterministic input signal
Inp   = np.squeeze(np.ones((N,1)))/np.sqrt(N) 
Ipert = np.squeeze(np.ones((N,1)))/np.sqrt(N) 
# Ipert[NE:]=0
Ipert[:NE]=0
tt = np.linspace(0,200,1000)

# %%
def case1eigv_complex(x,J,g,NE,NI,c,tau,tau_fixed):
    N = NE+NI
    ji,je = g*J,J 
    JE,JI = je*c*NE, ji*c*NI 
    nvec, mvec = np.zeros((N,1)), np.ones((N,1))
    nvec[:NE,0], nvec[NE:,0] = JE/NE, -JI/NI
    lambda_complex = x[0]+1j*x[1]
    tauc = np.abs(tau_fixed)
    zee = J**2*c*(1-c)*NE*tauc-g*J**2*c*(1-c)*NI*tau
    zii = -g*J**2*c*(1-c)*NE*tau+g**2*J**2*c*(1-c)*NI*tauc 
    # print((NE*zee+NI*zii)*4+(JE-JI)**2)
    z2lambda = np.zeros((N,N),dtype=complex)
    z2lambda[:,:NE], z2lambda[:,NE:] = zee/lambda_complex**2,zii/lambda_complex**2 
    z2lambda = np.eye(N)-z2lambda 
    final = nvec.T@la.inv(z2lambda)@mvec 
    final = np.squeeze(final)
    return [final.real-x[0],final.imag-x[1]]

# %%
def processInput(tau_fixed):
    ### define the network parameters of the adjacency matrix
    N  = 1500
    Kt = int(N*0.2)
    J = 1/np.sqrt(N)*0.5  ### TODO: make sure this scalar with David&Stefano's paper
    J = 0.00325
    ntau   = 10
    trials = 3
    tau_series = np.linspace(0,0.15,ntau)#np.linspace(0.1,0.2,ntau)#


    g, gamma =  6, 1/4.0
    NE = int(N/(1+gamma))
    NI = N-NE#NE*gamma
    N  = NE+NI ### update 
    ALPHAE, ALPHAI = NE/N, NI/N
    KE, KI = int(Kt/(1+gamma)), int(Kt/(1+gamma)*gamma) ### fixed out-degree
    ce, ci = KE/NE, KI/NI
    # print('ce and ci:',ce,ci)
    ### assert that the differences between ce and ci are smaller than epsilon
    epsilon = 1E-2
    assert np.abs(ce-ci)<epsilon
    # assert ce==ci
    c = ce
    ji,je = g*J,J 
    ### define the network parameters of the diluted Gaussian matrix 
    ge, gi = np.sqrt(je**2*c*(1-c)*N), np.sqrt(ji**2*c*(1-c)*N) 
    hat_sigmae, hat_sigmai = np.sqrt(c*(1-c)), np.sqrt(c*(1-c))### standard deviation of the adjacency matrix
    sigmae,sigmai = np.sqrt(c*(1-c)*J**2*N), np.sqrt(c*(1-c)*(-g*J)**2*N)### with magnitude of the coupling
    JE,JI = je*c*NE, ji*c*NI 
    lambda0 = JE-JI 

        
    #### constant and deterministic input signal
    Inp   = np.squeeze(np.ones((N,1)))/np.sqrt(N) 
    Ipert = np.squeeze(np.ones((N,1)))/np.sqrt(N) 
    # Ipert[NE:]=0
    Ipert[:NE]=0
    tt = np.linspace(0,200,1000)
    
    ''' refresh'''
    ### EEII; EEEIII; EEEIII;EEII
    switch = 1
    ## arrays to store results
    ## norml0_series: norm of left eigenvector(deltaliri = 1)
    eigvchn_series, eigrvec_series, eiglvec_series = np.zeros((trials,ntau,N),dtype=complex), np.zeros((trials,ntau,N,2)), np.zeros((trials,ntau,N,2))
    eiglvec0_series, norml0_series = np.zeros((trials,ntau,N,2)), np.zeros((trials,ntau+1,N)) 
    ### also have the reconstructed left and right eigenvectors 
    eigrvec_series_rec, eiglvec_series_rec = np.zeros((trials,ntau,N,2)), np.zeros((trials,ntau,N,2))
    htau = tau_series[1]-tau_series[0]
    ### simulation using the low-rank framework
    firing_rateeq = np.zeros((trials,ntau,N))
    ### recording dynamics
    lowrank_eq, lowrank_eq_num = np.zeros((trials,ntau,2)), np.zeros((trials,ntau,N))
    ovs_inplr, ovs_inplr_num   = np.zeros((trials,ntau,2)), np.zeros((trials,ntau,2))
    ovs_inplr_div, ovs_inplr_div_num = np.zeros((trials,ntau,2)), np.zeros((trials,ntau,2))
    contributions_lr, contributions_lr_num = np.zeros((trials,ntau,2,2)), np.zeros((trials,ntau,2,2))### rank, population
    intg_mean_series  = np.zeros((trials,ntau,N,2,2))### rank2 and population2
    ### mean connectivity
    nvec, mvec = np.zeros((N,1)), np.ones((N,1))
    nvec[:NE,0], nvec[NE:,0] = N*JE/NE, -N*JI/NI
    Jbar = mvec@nvec.T/N 
    ## TEST THE EIGENVALUES OF THE MEAN MATRIX 
    eigvJ0, eigvecJ0 = la.eig(Jbar)
    ### mean left and right eigenvectors
    leigvec0, reigvec0 = np.zeros((N,N)), np.zeros((N,N))
    norm_left = np.zeros(2)
    ## first eigenvector
    leigvec0[:,0], reigvec0[:,0] = nvec[:,0]/(JE-JI)/np.sqrt(N), mvec[:,0]/np.sqrt(N)
    norm_left[0]  = la.norm(leigvec0[:,0])
    leigvec0[:,0] = leigvec0[:,0]/norm_left[0]
    norml0_series[:,0,0] = np.sum(leigvec0[:,0]*reigvec0[:,0])
    ## second eigenvector
    kk = np.sqrt(NE*JI**2+NI*JE**2)
    reigvec0[:NE,1], reigvec0[NE:,1] = JI/kk,JE/kk 
    leigvec0[:NE,1], leigvec0[NE:,1] = -kk/(JE-JI)/NE,kk/(JE-JI)/NI 
    norm_left[1]     = la.norm(leigvec0[:,1])
    leigvec0[:,1] = leigvec0[:,1]/norm_left[1]
    norml0_series[:,0,1] = np.sum(leigvec0[:,1]*reigvec0[:,1])
    outerproduct  = np.sum(leigvec0[:,0]*reigvec0[:,1])#*norm_left
    
    ''' run  '''
    Inp   = np.squeeze(np.ones((N,1)))/np.sqrt(N) 
    firing_ratepert = np.zeros((trials,ntau,N))
    temporal_fr = np.zeros((trials,ntau,N,1000))
    temporal_perturb_fr = np.zeros((trials,ntau,N,1000))
    eiglvec0norm_series= np.zeros((trials,ntau,N,2))
    leig0mean_series = np.zeros((trials,ntau,N,2))
    leig0pre_series = np.zeros((trials,ntau,N,2))

    norm_4rvec_series, norm_4lvec_series = np.zeros((trials,ntau,2)),np.zeros((trials,ntau,2))
    norm_4lvec_series_ = np.zeros((trials,ntau,2))
    flag_run = True
    mode = 'heterogeneous'#'nonneg'#'normal'#
    for ktrial in range(trials):
        xr      = iidGaussian([0,1/np.sqrt(N)],[N,N])
        xrec    = iidGaussian([0,1/np.sqrt(N)],[N,N])
        ### zscore
        xr   = stats.zscore(xr.flatten())
        xr   = xr*1/np.sqrt(N)
        xr   = np.reshape(xr,(N,N))
        ### zscore
        xrec = stats.zscore(xrec.flatten())
        xrec = xrec*1/np.sqrt(N)
        xrec = np.reshape(xrec,(N,N))
        # ### zscore
        # nsample = 20
        chneta  = iidGaussian([0,1/np.sqrt(N)],[N,6])
        chneta[:,0] = stats.zscore(chneta[:,0])
        chneta[:,0] *=(1/np.sqrt(N))
        
        chneta[:,1] = stats.zscore(chneta[:,1])
        chneta[:,1] *=(1/np.sqrt(N))
        
        chneta[:,2] = stats.zscore(chneta[:,2])
        chneta[:,2] *=(1/np.sqrt(N))  
        
        ### ---------------------
        intg_ml, intg_mr = leigvec0.copy(), reigvec0.copy()
        z_pre = np.zeros((N,N))
        for it, tau in enumerate(tau_series[:]):
            tau = -tau
            if mode == 'heterogeneous':
                ''' heterogeneous '''
                aee,aii = np.sqrt(np.abs(tau_fixed)),np.sqrt(np.abs(tau_fixed))
                aei,aie = np.sqrt(np.abs(tau)),np.sqrt(np.abs(tau))
                zrowe = np.repeat(np.reshape(chneta[:,0],(1,-1)),N,axis=0)
                zcole = np.repeat(np.reshape(chneta[:,0],(-1,1)),N,axis=1)
                
                zrowi = np.repeat(np.reshape(chneta[:,1],(1,-1)),N,axis=0)
                zcoli = np.repeat(np.reshape(chneta[:,1],(-1,1)),N,axis=1)
                
                if tau<0:
                    sgnn = -1
                else:
                    sgnn = 1
            
                ### zee and zii    
                z_ee = np.zeros((NE,NE))
                z_ee = aee*(zcole[:NE,:NE])+aee*(zrowe[:NE,:NE])+aei*(zrowi[:NE,:NE])+np.sqrt(1-aee**2-aee**2-aei**2)*xr[:NE,:NE]
                
                z_ii = np.zeros((NI,NI))
                z_ii = aii*(zcoli[NE:,NE:])+aie*(zrowe[NE:,NE:])+aii*(zrowi[NE:,NE:])+np.sqrt(1-aii**2-aie**2-aii**2)*xr[NE:,NE:]
                
                z_ei = np.zeros((NE,NI))
                z_ei = sgnn*aei*(zcoli[:NE,NE:])+aie*(zrowe[:NE,NE:])+aii*(zrowi[:NE,NE:])+np.sqrt(1-aei**2-aie**2-aii**2)*xr[:NE,NE:]
                
                z_ie = np.zeros((NI,NE))
                z_ie = sgnn*aie*(zcole[NE:,:NE])+aee*(zrowe[NE:,:NE])+aei*(zrowi[NE:,:NE])+np.sqrt(1-aie**2-aee**2-aei**2)*xr[NE:,:NE]
                
                zr = np.zeros((N,N))
                zr[:NE,:NE] = z_ee.copy()
                zr[NE:,NE:] = z_ii.copy()
                zr[:NE,NE:] = z_ei.copy()
                zr[NE:,:NE] = z_ie.copy()
            
                ### E-I matrix 
                zr[:,:NE],zr[:,NE:] = zr[:,:NE]*ge,zr[:,NE:]*(-gi)   
            elif mode == 'normal':
                ''' Normal'''
                a    = np.sqrt(tau)
                zrow = a*np.repeat(np.reshape(chneta[:,0],(1,-1)),N,axis=0)
                zcol = a*np.repeat(np.reshape(chneta[:,0],(-1,1)),N,axis=1)
                gammarec = a*xrec-a*xrec.copy().T
                ### adjacency matrix
                zr   = zrow.copy()+zcol.copy()+np.sqrt(1-2*tau)*xr### without considering the reciprocal term
                ### E-I matrix 
                zr[:,:NE],zr[:,NE:] = zr[:,:NE]*ge,zr[:,NE:]*(-gi) 
            elif mode == 'nonneg':
                ''' nonneg '''
                aee,aii = np.sqrt(np.abs(tau)),np.sqrt(np.abs(tau))
                aei,aie = 0,0
                zrowe = np.repeat(np.reshape(chneta[:,0],(1,-1)),N,axis=0)
                zcole = np.repeat(np.reshape(chneta[:,0],(-1,1)),N,axis=1)
                
                zrowi = np.repeat(np.reshape(chneta[:,1],(1,-1)),N,axis=0)
                zcoli = np.repeat(np.reshape(chneta[:,1],(-1,1)),N,axis=1)
                
                if tau<0:
                    sgnn = -1
                else:
                    sgnn = 1
            
                ### zee and zii    
                z_ee = np.zeros((NE,NE))
                z_ee = aee*(zcole[:NE,:NE])+aee*(zrowe[:NE,:NE])+aei*(zrowi[:NE,:NE])+np.sqrt(1-aee**2-aee**2-aei**2)*xr[:NE,:NE]
                
                z_ii = np.zeros((NI,NI))
                z_ii = aii*(zcoli[NE:,NE:])+aie*(zrowe[NE:,NE:])+aii*(zrowi[NE:,NE:])+np.sqrt(1-aii**2-aie**2-aii**2)*xr[NE:,NE:]
                
                z_ei = np.zeros((NE,NI))
                z_ei = sgnn*aei*(zcoli[:NE,NE:])+aie*(zrowe[:NE,NE:])+aii*(zrowi[:NE,NE:])+np.sqrt(1-aei**2-aie**2-aii**2)*xr[:NE,NE:]
                
                z_ie = np.zeros((NI,NE))
                z_ie = sgnn*aie*(zcole[NE:,:NE])+aee*(zrowe[NE:,:NE])+aei*(zrowi[NE:,:NE])+np.sqrt(1-aie**2-aee**2-aei**2)*xr[NE:,:NE]
                
                zr = np.zeros((N,N))
                zr[:NE,:NE] = z_ee.copy()
                zr[NE:,NE:] = z_ii.copy()
                zr[:NE,NE:] = z_ei.copy()
                zr[NE:,:NE] = z_ie.copy()
            
                ### E-I matrix 
                zr[:,:NE],zr[:,NE:] = zr[:,:NE]*ge,zr[:,NE:]*(-gi)      
        
            DELTA_Z = zr-z_pre
            hzr     = DELTA_Z.copy()
            ### generate J connectivity matrix
            Jchn = Jbar.copy()+zr.copy()
            ### full rank simulation
            xinit = np.squeeze(np.random.normal(0, 1E-2, (1, N)))
            xc_temporal = odesimulation(tt, xinit, Jchn, Inp)
            temporal_fr[ktrial,it,:,:] = np.squeeze(xc_temporal.T).copy()
            firing_rateeq[ktrial,it,:] = xc_temporal[-1,:].copy()
            ### perturbation 
            xpert = xc_temporal[-1,:].copy()
            xpert = xpert.reshape(-1,1)
            dtt =tt[1]-tt[0]
            xc_temporal_perturb = []
            for ttt in range(len(tt)):
                delta_x= -xpert + Jchn@xpert.reshape(-1,1)+Ipert.reshape(-1,1)+Inp.reshape(-1,1)
                xpert = delta_x*dtt+xpert 
                xc_temporal_perturb.append(xpert)
            firing_ratepert[ktrial,it,:] = xpert.copy().squeeze()
            temporal_perturb_fr[ktrial,it,:,:] = np.squeeze(np.array(xc_temporal_perturb).T)
            
            
            eigvchn, eigrvec = la.eig(Jchn)
            
            eigvchn_,eiglvec = la.eig(Jchn.copy().T)
            ### normalization
            reig  = np.squeeze(eigrvec[:,:].copy())
            leig0 = np.squeeze(eiglvec[:,:].copy()) 
            normval = np.sum(reig.copy()*leig0.copy(),axis=0)
            norml0_series[ktrial,it+1,:] = normval.copy() ### normalization factor shift right 1byte
            normval = np.repeat(np.reshape(normval,(1,N)),N,axis=0)
            leig    = leig0.copy()/normval.copy() ### left eigenvector normalization
            
            ### sort the eigenvalues by the real part
            if eigvchn[0].real>eigvchn[1].real:
                idxsort = [1,0]
            else:
                idxsort = [0,1]
            eigvchn[:2] = eigvchn[idxsort]
            reig[:,:2] = reig[:,idxsort]
            leig[:,:2] = leig[:,idxsort]
            leig0[:,:2] = leig0[:,idxsort]
            
            if np.mean(reig[:NE,0])*np.mean(reigvec0[:NE,0])<0:
                reig[:,0]*=-1
                leig[:,0]*=-1
                leig0[:,0]*=-1
            if np.mean(reig[:NE,1])<0: ### the second rank-1 component is negative
                reig[:,1]*=-1
                leig[:,1]*=-1
                leig0[:,1]*=-1
                
            ### numerical low-rank approximation 
            ov_inp_lowrank,ov_inp_lowrank_div = np.zeros(2),np.zeros(2)
            vec_lowrank_contribution = np.zeros((N,2))
            ### linear response theory approximation 
            for i in range(2):
                ov_inp_lowrank[i]=np.sum(leig[:,i]*Inp[:])*eigvchn[i]   
                ov_inp_lowrank_div[i]= ov_inp_lowrank[i]/(1-eigvchn[i])
            Equilibrium_lowrank_outliers = np.reshape(Inp.copy(),(N,1))
            # print('EQ shape:',np.shape(Equilibrium_lowrank_outliers))
            for i in range(2):
                vec_lowrank_contribution[:,i] = ov_inp_lowrank_div[i]*reig[:,i]
                Equilibrium_lowrank_outliers += np.reshape(vec_lowrank_contribution[:,i].copy(),(N,1))
                #### REDUCE TO 2 POPULATION, THEREFORE RANK, POPULATION
                contributions_lr_num[ktrial,it,i,0] = np.mean(vec_lowrank_contribution[:NE,i])
                contributions_lr_num[ktrial,it,i,1] = np.mean(vec_lowrank_contribution[NE:,i])
            lowrank_eq_num[ktrial,it,:] = np.squeeze(Equilibrium_lowrank_outliers.copy())
            ovs_inplr_num[ktrial,it,:],ovs_inplr_div_num[ktrial,it,:] = ov_inp_lowrank.copy(),ov_inp_lowrank_div.copy() 
                
            
            
            eigvchn_series[ktrial,it,:]    = eigvchn.copy()#eigvw_norm.copy()#
            eigrvec_series[ktrial,it,:,:]  = reig[:,:2].copy()#eigvecw_norm.copy()#
            eiglvec_series[ktrial,it,:,:]  = leig[:,:2].copy()#eigvect_norm.copy()#
            eiglvec0_series[ktrial,it,:,:] = leig0[:,:2].copy()#eigvect_norm.copy()#
            for iii in range(2):
                eiglvec0norm_series[ktrial,it,:,iii] = leig0[:,iii].copy()/normval[iii,iii]
                
            hzr_u = hzr.copy()
            # hzr_u[:NE,NE:], hzr_u[NE:,:NE]=0,0
            DeltaZ2 = hzr_u@hzr_u    ### used to correct        
            if it<1:
                lvec, rvec = np.squeeze(leigvec0[:,:2]),np.squeeze(reigvec0[:,:2])
                # eigeng     = np.squeeze(eigvchn_series[ktrial,it,:2].copy())
                eigeng     = np.array([JE-JI,0])
                lvec_mean, rvec_mean = lvec.copy(), rvec.copy()
            else:
                lvec, rvec = np.squeeze(eiglvec0_series[ktrial,it-1,:,:2]),np.squeeze(eigrvec_series[ktrial,it-1,:,:2]) ### use the previous eigenvector as the initial condition   
                eigeng = np.squeeze(eigvchn_series[ktrial,it-1,:2].copy())
                ### conditioned mean  
                lvec_mean, rvec_mean = lvec.copy(), rvec.copy()
                lvec_mean[:NE,:],lvec_mean[NE:,:]=np.mean(lvec_mean[:NE,:],axis=0),np.mean(lvec_mean[NE:,:],axis=0)
                rvec_mean[:NE,:],rvec_mean[NE:,:]=np.mean(rvec_mean[:NE,:],axis=0),np.mean(rvec_mean[NE:,:],axis=0)
                
            ### get the appropriate normalization factor
            norm_for_lvec, norm_for_rvec = np.zeros(2),np.zeros(2)
            if it==0:
                hzr_u = xr.copy()
                with_chn = 0
            else:
                hzr_u = hzr.copy()
                with_chn = 1
            
            if mode == 'nonneg':
                truncc = 3
            else:
                truncc = 1
            norm_rvec_temp, norm_lvec_temp = np.zeros((N,2)), np.zeros((N,2))
            for i in range(2):
                rvec_n = np.reshape(rvec[:,i],(-1,1)) + (hzr_u)@np.reshape(rvec[:,i].copy(),(-1,1))/(eigeng[i])#np.real(eigeng[i])
                lvec_n = np.reshape(lvec[:,i],(-1,1)) + (hzr_u).T@np.reshape(lvec[:,i].copy(),(-1,1))/(eigeng[i])#np.real(eigeng[i])

                '''mean connectivity'''
                current_eigv = eigvchn_series[ktrial,it,i].copy()
                if it==0:
                    intg_ml[:NE,i],intg_ml[NE:,i] = np.mean(leig[:NE,i])*current_eigv, np.mean(leig[NE:,i])*current_eigv
                    intg_mr[:NE,i],intg_mr[NE:,i] = np.mean(reig[:NE,i]), np.mean(reig[NE:,i])
                    leig0mean_series[ktrial,it,:,i] =leig0[:,i]/norml0_series[ktrial,it+1,i]
                    ''' otherwise no correlation can be calculated '''
                    norm_rvec_temp[:,i] = np.squeeze(reig[:,i])
                    norm_lvec_temp[:,i] = np.squeeze(leig[:,i])*current_eigv
                elif it < truncc and i==1:### alway >0
                    ### original values are obtained numerically
                    intg_ml[:NE,i],intg_ml[NE:,i] = np.mean(leig[:NE,i])*current_eigv, np.mean(leig[NE:,i])*current_eigv
                    intg_mr[:NE,i],intg_mr[NE:,i] = np.mean(reig[:NE,i]), np.mean(reig[NE:,i])
                    leig0mean_series[ktrial,it,:,i] =leig0[:,i]/norml0_series[ktrial,it+1,i]
                    ''' otherwise no correlation can be calculated '''
                    norm_rvec_temp[:,i] = np.squeeze(reig[:,i])
                    norm_lvec_temp[:,i] = np.squeeze(leig[:,i])*current_eigv
                else:
                    eigenvalue_u = (current_eigv)#np.real(current_eigv)
                    eigenvalue_um = (eigeng[i]) #np.real(eigeng[i]) 
                    
                    ### norm_for_rvec and norm_for_lvec 
                    rmean_tmp =np.reshape(rvec_mean[:,i].copy(),(-1,1)) + np.reshape(with_chn*(DeltaZ2@np.reshape(rvec_mean[:,i],(-1,1)))/(eigenvalue_um)**2,(N,1))### mean-shifting #np.real(eigenvalue_um)**2,(N,1))### mean-shifting 
                    rvec_n[:NE,0] = rvec_n[:NE,0] - np.mean(rvec_n[:NE,0])+rmean_tmp[:NE,0]
                    rvec_n[NE:,0] = rvec_n[NE:,0] - np.mean(rvec_n[NE:,0])+rmean_tmp[NE:,0]

                    lmean_tmp = np.reshape(lvec_mean[:,i].copy(),(-1,1)) + np.reshape(with_chn*np.reshape(lvec_mean[:,i],(1,-1))@DeltaZ2/(eigeng[i])**2,(N,1))#np.real(eigeng[i])**2,(N,1))### mean-shifting 
                    # print('lvec mean E:', lvec_mean[::NE,i])
                    if i==1:
                        shiftlvec = np.reshape(with_chn*np.reshape(lvec_mean[:,i],(1,-1))@DeltaZ2/(eigeng[i])**2,(N,1))##np.real(eigeng[i])**2,(N,1))### mean-shifting 

                    # print('lvec mean E:', np.mean(lvec_n[:NE,0]),np.mean(lvec_n[NE:,0]))
                    lvec_n[:NE,0] = lvec_n[:NE,0] - np.mean(lvec_n[:NE,0])+lmean_tmp[:NE,0]
                    lvec_n[NE:,0] = lvec_n[NE:,0] - np.mean(lvec_n[NE:,0])+lmean_tmp[NE:,0]
                    
                    norm_for_rvec[i] = la.norm(rvec_n) ### normalization factor 
                    norm_rvec_n = np.reshape(rvec_n.copy(),(-1,1))/norm_for_rvec[i] 
                    norm_rvec_temp[:,i]=np.squeeze(norm_rvec_n.copy())
                    
                    norm_for_lvec[i] = np.squeeze(np.reshape(lvec_n,(1,-1))@np.reshape(norm_rvec_temp[:,i],(-1,1)))#la.norm(lvec_n)#norml0_series[ktrial,it,i]#
                    norm_lvec_n = np.reshape(lvec_n.copy(),(-1,1))/norm_for_lvec[i] 
                    ### need to be re-normalized
                    norm_lvec_temp[:,i]  = np.squeeze(norm_lvec_n.copy())
                    norm_lvec_temp[:,i] = norm_lvec_temp[:,i] *eigvchn_series[ktrial,it,i]#eigvchn_series[ktrial,it,i].real
                    
                    ### more simplified version using lvec_mean and rvec_mean
                    intg_ml[:,i] = lmean_tmp[:,0]/norm_for_lvec[i]*eigenvalue_u
                    leig0mean_series[ktrial,it,:,i] =lmean_tmp[:,0]/norm_for_lvec[i]
                    leig0pre_series[ktrial,it,:,i] =np.reshape(with_chn*np.reshape(lvec_mean[:,i],(1,-1))@DeltaZ2/(eigeng[i])**2,(N,1))[:,0]##np.real(eigeng[i])**2,(N,1))[:,0]#*eigenvalue_u#
                    intg_mr[:,i] = rmean_tmp[:,0]/norm_for_rvec[i]
                    
                    tilden = np.reshape(lvec_n.copy(),(-1,1))/norml0_series[ktrial,it,i]
                    norm_4lvec_series_[ktrial,it,i]=np.squeeze(np.reshape(tilden,(1,-1))@np.reshape(rvec_n.copy(),(-1,1))/norm_for_rvec[i])
                    
                    norm_4rvec_series[ktrial,it,i],norm_4lvec_series[ktrial,it,i]=norm_for_rvec[i],norm_for_lvec[i]
                    # if i==0:
                    #     print(i,'should be the same',norm_4lvec_series[ktrial,it,i]/norml0_series[ktrial,it,i],norm_4lvec_series_[ktrial,it,i])
                    
                ## @YS 17 Nov, no matter what the variable it is.
                ### modify the mean of the elements on the left and right eigenvectors
                norm_rvec_temp[:NE,i] -= np.mean(norm_rvec_temp[:NE,i])
                norm_rvec_temp[NE:,i] -= np.mean(norm_rvec_temp[NE:,i])
                norm_rvec_temp[:NE,i] += np.mean(intg_mr[:NE,i])
                norm_rvec_temp[NE:,i] += np.mean(intg_mr[NE:,i])
                norm_lvec_temp[:NE,i] -= np.mean(norm_lvec_temp[:NE,i])
                norm_lvec_temp[NE:,i] -= np.mean(norm_lvec_temp[NE:,i])
                norm_lvec_temp[:NE,i] += np.mean(intg_ml[:NE,i])
                norm_lvec_temp[NE:,i] += np.mean(intg_ml[NE:,i])
                
                ### also record the reconstructed eigenvectors
                eigrvec_series_rec[ktrial,it,:,i] = norm_rvec_temp[:,i].copy()
                eiglvec_series_rec[ktrial,it,:,i] = norm_lvec_temp[:,i].copy()
                
                intg_mean_series[ktrial,it,:,i,0] = np.reshape(intg_ml[:,i],(N,))
                intg_mean_series[ktrial,it,:,i,1] = np.reshape(intg_mr[:,i],(N,))
                
            
            z_pre = zr.copy()
            flag_run = True
            ### theoretically compute the equilibrium population-averaged firing rate
            an = np.zeros((2,2),dtype=complex)
            am = np.zeros((2,2),dtype=complex) ## population X rank    
            for ir in range(2):
                an[0,ir] = np.mean(intg_ml[:NE,ir])
                an[1,ir] = np.mean(intg_ml[NE:,ir])
                am[0,ir] = np.mean(intg_mr[:NE,ir])
                am[1,ir] = np.mean(intg_mr[NE:,ir])
                
            ### overlap sum
            overlap_inp = np.zeros(2,dtype=complex) 
            for ir in range(2):
                overlap_inp[ir] = (NE*an[0,ir]*Inp[0]+NI*an[1,ir]*Inp[-1])
                ovs_inplr[ktrial,it,ir] = overlap_inp[ir] ### recording_theory
                overlap_inp[ir]/= (1.0-eigvchn[ir])
                ovs_inplr_div[ktrial,it,ir] = overlap_inp[ir] ### recording theory
            eq_fr = np.zeros(2)
            eq_fr[0],eq_fr[1] = Inp[0],Inp[-1]
            for ir in range(2): ### rank two 
                ### record 
                contributions_lr[ktrial,it,ir,0] = overlap_inp[ir]*am[0,ir] ### excitatory population
                contributions_lr[ktrial,it,ir,1] = overlap_inp[ir]*am[1,ir] ### inhibitory population
                eq_fr[0] += am[0,ir]*overlap_inp[ir]
                eq_fr[1] += am[1,ir]*overlap_inp[ir]
            lowrank_eq[ktrial,it,:] = eq_fr.copy() 
            
    eigvchn_real = eigvchn_series[:,-1,0].real.copy()
    ### sorting eigvchn_real 
    idx = np.argsort(eigvchn_real)
    idx_eff = idx[3:-3]#idx

    ### solve x 
    eigvchn_theo = np.zeros((ntau,2),dtype=complex)
    for it, tau in enumerate(tau_series):
        if it<=-1:
            continue
        try:
            x = fsolve(case1eigv_complex,[np.mean(eigvchn_series[idx_eff,it,1].real),np.mean(eigvchn_series[idx_eff,it,1].imag)],args=(J,g,NE,NI,c,tau,tau_fixed))
            eigvchn_theo[it,1] = x[0]+1j*x[1] 
        except:
            eigvchn_theo[it,1] = np.nan
        
        try:
            x = fsolve(case1eigv_complex,[np.mean(eigvchn_series[idx_eff,it,0].real),np.mean(eigvchn_series[idx_eff,it,0].imag)],args=(J,g,NE,NI,c,tau,tau_fixed))
            eigvchn_theo[it,0] = x[0]+1j*x[1] 
        except:
            eigvchn_theo[it,0] = np.nan
            
    ### compute the trial averaged mean 
    mean_reigvec_series = np.zeros((trials,ntau,2,2)) ##rank, pop
    mean_leigvec_series = np.zeros((trials,ntau,2,2)) ##rank, pop
    ### numerical
    mean_reigvec_num_series = np.zeros((trials,ntau,2,2))
    mean_leigvec_num_series = np.zeros((trials,ntau,2,2))
    thl = 15
    ths = 15
    for ktrial in range(trials):
        for it in range(ntau):
            if it<2:
                threshold = ths
            else:
                threshold = thl
            for ir in range(2):
                mean_reigvec_series[ktrial,it,ir,0] = np.mean(intg_mean_series[ktrial,it,:NE,ir,1],axis=0)
                
                mean_reigvec_series[ktrial,it,ir,1] = np.mean(intg_mean_series[ktrial,it,NE:,ir,1],axis=0)
                
                mean_leigvec_series[ktrial,it,ir,0] = np.mean(intg_mean_series[ktrial,it,:NE,ir,0],axis=0)
                mean_leigvec_series[ktrial,it,ir,1] = np.mean(intg_mean_series[ktrial,it,NE:,ir,0],axis=0)

                    
    ### for the numerical 
    for ktrial in range(trials):
        for it in range(ntau):
            if it<2:
                threshold = ths
            else:
                threshold = thl
            for ir in range(2):
                mean_reigvec_num_series[ktrial,it,ir,0] = np.mean(eigrvec_series[ktrial,it,:NE,ir])
                mean_reigvec_num_series[ktrial,it,ir,1] = np.mean(eigrvec_series[ktrial,it,NE:,ir])
                mean_leigvec_num_series[ktrial,it,ir,0] = np.mean(eiglvec_series[ktrial,it,:NE,ir])
                mean_leigvec_num_series[ktrial,it,ir,1] = np.mean(eiglvec_series[ktrial,it,NE:,ir])

                
    ## select the middle 30 values 
    kktrial = np.arange(trials)
    cuts = 6
    for it in range(ntau):
        for ir in range(2):
            ### only keep the middle 30 values of mean_reigvec_series[:,it,ir,0/1], osrt mean_reigvec_series[:,it,ir,0]
            idxsort = np.argsort(mean_reigvec_series[:,it,ir,0].real)
            idxnan = np.append(idxsort[:cuts],idxsort[-cuts:]) 
            mean_reigvec_series[idxnan,it,ir,0] = np.nan 
            idxnan = np.where(np.abs(mean_reigvec_series[:,it,ir,0].real)>threshold)[0]
            mean_reigvec_series[idxnan,it,ir,0] = np.nan
            
            idxsort = np.argsort(mean_reigvec_series[:,it,ir,1].real)
            idxnan = np.append(idxsort[:cuts],idxsort[-cuts:])
            mean_reigvec_series[idxnan,it,ir,1] = np.nan
            idxnan = np.where(np.abs(mean_reigvec_series[:,it,ir,1].real)>threshold)[0]
            mean_reigvec_series[idxnan,it,ir,1] = np.nan
            
            
            idxsort = np.argsort(mean_leigvec_series[:,it,ir,0].real)
            idxnan = np.append(idxsort[:cuts],idxsort[-cuts:])
            mean_leigvec_series[idxnan,it,ir,0] = np.nan
            idxnan = np.where(np.abs(mean_leigvec_series[:,it,ir,0].real)>threshold)[0]
            mean_leigvec_series[idxnan,it,ir,0] = np.nan
                
            
            idxsort = np.argsort(mean_leigvec_series[:,it,ir,1].real)
            idxnan = np.append(idxsort[:cuts],idxsort[-cuts:])
            mean_leigvec_series[idxnan,it,ir,1] = np.nan
            idxnan = np.where(np.abs(mean_leigvec_series[:,it,ir,1].real)>threshold)[0]
            mean_leigvec_series[idxnan,it,ir,1] = np.nan
            
            idxsort = np.argsort(mean_reigvec_num_series[:,it,ir,0].real)
            idxnan = np.append(idxsort[:cuts],idxsort[-cuts:]) 
            mean_reigvec_num_series[idxnan,it,ir,0] = np.nan 
            idxnan = np.where(np.abs(mean_reigvec_num_series[:,it,ir,0].real)>threshold)[0]
            mean_reigvec_num_series[idxnan,it,ir,0] = np.nan
            
            idxsort = np.argsort(mean_reigvec_num_series[:,it,ir,1].real)
            idxnan = np.append(idxsort[:cuts],idxsort[-cuts:])
            mean_reigvec_num_series[idxnan,it,ir,1] = np.nan
            idxnan = np.where(np.abs(mean_reigvec_num_series[:,it,ir,1].real)>threshold)[0]
            mean_reigvec_num_series[idxnan,it,ir,1] = np.nan
            
            idxsort = np.argsort(mean_leigvec_num_series[:,it,ir,0].real)
            idxnan = np.append(idxsort[:cuts],idxsort[-cuts:])
            mean_leigvec_num_series[idxnan,it,ir,0] = np.nan
            idxnan = np.where(np.abs(mean_leigvec_num_series[:,it,ir,0].real)>threshold)[0]
            mean_leigvec_num_series[idxnan,it,ir,0] = np.nan
            
            idxsort = np.argsort(mean_leigvec_num_series[:,it,ir,1].real)
            idxnan = np.append(idxsort[:cuts],idxsort[-cuts:])
            mean_leigvec_num_series[idxnan,it,ir,1] = np.nan
            idxnan = np.where(np.abs(mean_leigvec_num_series[:,it,ir,1].real)>threshold)[0]
            mean_leigvec_num_series[idxnan,it,ir,1] = np.nan
            
    mean_rvec = np.zeros((ntau,2,2))
    mean_lvec = np.zeros((ntau,2,2))

    for it in range(ntau):
        for ir in range(2):
            for ip in range(2):
                mean_rvec[it,ir,ip] = np.nanmean(mean_reigvec_series[:,it,ir,ip])
                mean_lvec[it,ir,ip] = np.nanmean(mean_leigvec_series[:,it,ir,ip])
                
    ### compute the theoretical response function 
    rank = 2
    response_func_contribution = np.zeros((ntau,rank,2))
    ### compute the theoretical response function 
    response_func = np.zeros((trials,ntau,2))
    for it in range(ntau):
        response_func[:,it,1] = 1
        if it>=0:
            if eigvchn_theo[it,1].real>0 and ~np.isnan(eigvchn_theo[it,1]):
                for ir in range(2):
                    response_func[:,it,0] += (NI*(mean_lvec[it,ir,1])*(mean_rvec[it,ir,0]))/(1-eigvchn_theo[it,ir].real)
                    response_func[:,it,1] += (NI*(mean_lvec[it,ir,1])*(mean_rvec[it,ir,1]))/(1-eigvchn_theo[it,ir].real)
                    
                    response_func_contribution[it,ir,0] = (NI*(mean_lvec[it,ir,1])*(mean_rvec[it,ir,0]))/(1-eigvchn_theo[it,ir].real)
                    response_func_contribution[it,ir,1] = (NI*(mean_lvec[it,ir,1])*(mean_rvec[it,ir,1]))/(1-eigvchn_theo[it,ir].real)
                    
            else:
                for ir in range(1):
                    response_func[:,it,0] += (NI*(mean_lvec[it,ir,1])*(mean_rvec[it,ir,0]))/(1-eigvchn_theo[it,ir].real)
                    response_func[:,it,1] += (NI*(mean_lvec[it,ir,1])*(mean_rvec[it,ir,1]))/(1-eigvchn_theo[it,ir].real)
                    
                    response_func_contribution[it,ir,0] = (NI*(mean_lvec[it,ir,1])*(mean_rvec[it,ir,0]))/(1-eigvchn_theo[it,ir].real)
                    response_func_contribution[it,ir,1] = (NI*(mean_lvec[it,ir,1])*(mean_rvec[it,ir,1]))/(1-eigvchn_theo[it,ir].real)
            
                    
        else:
            for ir in range(1):
                response_func[:,it,0] += (NI*(mean_lvec[it,ir,1])*(mean_rvec[it,ir,0]))/(1-eigvchn_theo[it,ir].real)
                response_func[:,it,1] += (NI*(mean_lvec[it,ir,1])*(mean_rvec[it,ir,1]))/(1-eigvchn_theo[it,ir].real)
                
                response_func_contribution[it,ir,0] = (NI*(mean_lvec[it,ir,1])*(mean_rvec[it,ir,0]))/(1-eigvchn_theo[it,ir].real)
                response_func_contribution[it,ir,1] = (NI*(mean_lvec[it,ir,1])*(mean_rvec[it,ir,1]))/(1-eigvchn_theo[it,ir].real)

    ### 
    meanfr_eq = np.zeros((trials,ntau,2))
    meanfr_pert = np.zeros((trials,ntau,2))
    meanfr_eq[:,:,0] = np.mean(firing_rateeq[:,:,:NE],axis=2)
    meanfr_eq[:,:,1] = np.mean(firing_rateeq[:,:,NE:],axis=2)
    ### same for pydll Creates ()
    meanfr_pert[:,:,0] = np.mean(firing_ratepert[:,:,:NE],axis=2)
    meanfr_pert[:,:,1] = np.mean(firing_ratepert[:,:,NE:],axis=2)
    ### numerical response function 
    response_func_num = np.zeros((trials,ntau,2))
    for ktrial in range(trials):
        for it in range(ntau):
            response_func_num[ktrial,it,0]=(meanfr_pert[ktrial,it,0]-meanfr_eq[ktrial,it,0])/Ipert[-1]
            response_func_num[ktrial,it,1]=(meanfr_pert[ktrial,it,1]-meanfr_eq[ktrial,it,1])/Ipert[-1]
            
    #### delete the largest 3 and the smallest 3 from response_func_num
    for it in range(ntau):
        idxsort = np.where(np.abs(response_func_num[:,it,0].copy())>10)
        idxnan = (idxsort) 
        response_func_num[idxnan,it,0] = np.nan 
        
        
        idxsort = np.where(np.abs(response_func_num[:,it,1].copy())>10)
        idxnan = (idxsort) 
        response_func_num[idxnan,it,1] = np.nan 
        
    response_func_mean = np.nanmean(response_func[:,:,:],axis=0)
    response_func_num_mean = np.nanmean(response_func_num[:,:,:],axis=0)
    
    ### save eigvchn_theo, response_func_mean, response_func_num_mean, response_func_num with the outeridx 
    data_dict = dict()
    data_dict['eigvchn_theo'] = eigvchn_theo.copy()
    data_dict['response_func_mean'] = response_func_mean.copy()
    data_dict['response_func_num_mean'] = response_func_num_mean.copy()
    data_dict['response_func_num'] = response_func_num.copy()
    ### save the data
    ### file_name add outeridx 
    file_name = "E:/Dropbox/DailyWork/Allen_project/preparation4paper_Data/"+str(tau_fixed)+"_data.npz"
    np.savez(file_name,**data_dict)
        
        

# %%
num_cores = mp.cpu_count()

# %%
num_cores

# %%
items = tau_series[-2:].copy()

# %%
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    executor.map(processInput, items)



