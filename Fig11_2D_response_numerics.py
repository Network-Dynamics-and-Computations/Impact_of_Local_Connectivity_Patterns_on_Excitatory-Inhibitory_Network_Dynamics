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
trials = 30+6
tau_series = np.linspace(0,0.15,ntau)#np.linspace(0.1,0.2,ntau)#
response_map_2d = np.zeros((ntau,ntau,2))
response_map_num_2d= np.zeros((ntau,ntau,2))
response_map_num_trials_2d= np.zeros((ntau,ntau,trials,2))
eigvchn_theo_2d = np.zeros((ntau,ntau,2),dtype=complex)

for outeridx in range(ntau):
    tau_fixed = tau_series[outeridx]
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

    ## numerical 
    intg_ov_series    = np.zeros((trials,ntau,2))
    first_perturb_ov  = np.zeros((trials,ntau,2,2))
    first_perturb_ovP = np.zeros((trials,ntau,2,2,2)) ### rank and population

    intg_mean_series  = np.zeros((trials,ntau,N,2,2))### rank2 and population2
    intg_std_series   = np.zeros((trials,ntau,2,2))### rank, population
    intg_std_num_series = np.zeros((trials,ntau,2,2))### rank, population
    mean_shift_ov     = np.zeros((trials,ntau,2))
    latent_kappa_series = np.zeros((trials,ntau,2,2))
    intg_crossov_series = np.zeros((trials,ntau,2,2)) # rank, rank
    intg_crossovPop_series = np.zeros((trials,ntau,2,2,2)) # rank, rank, population
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
    ### mean connectivity
    nvec, mvec = np.zeros((N,1)), np.ones((N,1))
    nvec[:NE,0], nvec[NE:,0] = N*JE/NE, -N*JI/NI
    Jbar = mvec@nvec.T/N 
    ## TEST THE EIGENVALUES OF THE MEAN MATRIX 
    eigvJ0, eigvecJ0 = la.eig(Jbar)
    print('eigvJ0:',eigvJ0[0],' theory:',JE-JI)
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
    print('JE:',JE,'JI:',JI)

    # %%
    #### constant and deterministic input signal
    Inp   = np.squeeze(np.ones((N,1)))/np.sqrt(N) 
    Ipert = np.squeeze(np.ones((N,1)))/np.sqrt(N) 
    # Ipert[NE:]=0
    Ipert[:NE]=0
    tt = np.linspace(0,200,1000)

    # %%
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
        print('~~~~~~~~~~~trial:',ktrial,'~~~~~~~~~~~~~~~')
        # while (flag_run):  
        # while(-1): 
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
        print('Go run...........................')
        intg_ov  = np.zeros(2) ### rank
        intg_ovP = np.zeros((2,2,2)) ### rank, rank, population
        intg_ml, intg_mr = leigvec0.copy(), reigvec0.copy()
        mean_pre = np.array([JE-JI,0])
        mean_total_change = np.zeros(2)
        z_pre = np.zeros((N,N))
        for it, tau in enumerate(tau_series[:]):
            tau = -tau ### try to model the homogeneous scenario
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
        
            zr2 = zr@zr   
            print(tau_series[outeridx],'..tau:',tau)
            print('zee2:',np.mean((zr[:NE,:NE]@zr[:NE,:NE]).flatten())/NE/(je**2)/c/(1-c))
            print('zii2:',np.mean((zr[NE:,NE:]@zr[NE:,NE:]).flatten())/NI/(ji**2)/c/(1-c))
            
            print('zei2:',np.mean((zr[:NE,:NE]@zr[:NE,NE:]).flatten())/NE/(je*ji)/c/(1-c))
            print('zIE2:',np.mean((zr[:NE,NE:]@zr[NE:,:NE]).flatten())/NI/(ji*je)/c/(1-c))
            
            print('trial:',ktrial,'tau:',tau)
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
            print('line 151 eigenvalues:',eigvchn[:2])
            
            if np.mean(reig[:NE,0])*np.mean(reigvec0[:NE,0])<0:
                reig[:,0]*=-1
                leig[:,0]*=-1
                leig0[:,0]*=-1
            if np.mean(reig[:NE,1])<0: ### the second rank-1 component is negative
                reig[:,1]*=-1
                leig[:,1]*=-1
                leig0[:,1]*=-1  
            
            
            eigvchn_series[ktrial,it,:]    = eigvchn.copy()#eigvw_norm.copy()#
            eigrvec_series[ktrial,it,:,:]  = reig[:,:2].copy()#eigvecw_norm.copy()#
            eiglvec_series[ktrial,it,:,:]  = leig[:,:2].copy()#eigvect_norm.copy()#
            eiglvec0_series[ktrial,it,:,:] = leig0[:,:2].copy()#eigvect_norm.copy()#
            

    # %%
    '''case 2'''
    nvec, mvec = np.zeros((N,1)), np.ones((N,1))
    nvec[:NE,0], nvec[NE:,0] = JE/NE, -JI/NI
    def case1eigv_complex(x,J,g,NE,NI,c,tau,tau_fixed):
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

    ### solve x 
    eigvchn_theo = np.zeros((ntau,2),dtype=complex)
    for it, tau in enumerate(tau_series):
        if it<=-1:
            continue
        try:
            x = fsolve(case1eigv_complex,[np.mean(eigvchn_series[:,it,1].real),np.mean(eigvchn_series[:,it,1].imag)],args=(J,g,NE,NI,c,tau,tau_fixed))
            eigvchn_theo[it,1] = x[0]+1j*x[1] 
        except:
            eigvchn_theo[it,1] = np.nan
        
        try:
            x = fsolve(case1eigv_complex,[np.mean(eigvchn_series[:,it,0].real),np.mean(eigvchn_series[:,it,0].imag)],args=(J,g,NE,NI,c,tau,tau_fixed))
            eigvchn_theo[it,0] = x[0]+1j*x[1] 
        except:
            eigvchn_theo[it,0] = np.nan
        
        print('tau:',tau,'eigv:',eigvchn_theo[it,:])
        print('numerical:',np.mean(eigvchn_series[:,it,:2],axis=0))

    ### numerical
    mean_reigvec_num_series = np.zeros((trials,ntau,2,2))
    mean_leigvec_num_series = np.zeros((trials,ntau,2,2))
    mean_leig0vec_num_series = np.zeros((trials,ntau,2,2))
    thl = 15
    ths = 15                    
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


    # %%
    ### 
    meanfr_eq   = np.zeros((trials,ntau,2))
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

    # %%
    ### compare response_func with response_func_num, average across the first dimension 
    dtau = tau_series[1]-tau_series[0]
    response_func_num_mean = np.nanmean(response_func_num[:,:,:],axis=0)

    # %%
    epsilon = 1e-6
    eigvchn_theo_2d[outeridx,:,:] = eigvchn_theo.copy()
    for it in range(ntau):
        if np.abs(eigvchn_theo[it,0].imag)>epsilon or np.abs(eigvchn_theo[it,1].imag)>epsilon:
            response_map_num_2d[outeridx,it,0] = np.nan
            response_map_num_2d[outeridx,it,1] = np.nan
            response_map_num_trials_2d[outeridx,it,:,:] = np.nan
        else:
            response_map_num_2d[outeridx,it,0] = response_func_num_mean[it,0]
            response_map_num_2d[outeridx,it,1] = response_func_num_mean[it,1]
            response_map_num_trials_2d[outeridx,it,:,:] = response_func_num[:,it,:]

    # %%
    print('numerical:',response_map_num_2d[outeridx,:,1])
