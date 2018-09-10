#!/usr/bin/env python

#import matplotlib.pyplot as plt
import numpy as np
#import scipy as sp
import math as mt
#import getopt
import sys
#direcciones
#para NERSC

##pre-reconstruction
inputCUTE='/global/cscratch1/sd/mmagana/EBOSS_ELG/v2.0/prerecon/'  #'/alicefs/marvam_g/bizb_a/eBOSS_ELG/mocks/mocks_rdzw/'  #'/alicefs/marvam_g/bizb_a/eBOSS_ELG/mocks/' 
outputCUTE_S='/global/cscratch1/sd/mmagana/EBOSS_ELG/v2.0/prerecon/' #'/scratch2/scratchdirs/izamar/output/prerecon/' #'/alicefs/marvam_g/bizb_a/eBOSS_ELG/output_ELG_mocks_rdzw/'  #'/alicefs/marvam_g/bizb_a/eBOSS_ELG/output_ELG_mocks/'
outputCUTE_N='/scratch2/scratchdirs/izamar/output/prerecon/'
randomCUTE='/global/cscratch1/sd/mmagana/EBOSS_ELG/v2.0/prerecon/'
#'/alicefs/marvam_g/bizb_a/eBOSS_ELG/mocks/mocks_rdzw/'  #'/alicefs/marvam_g/bizb_a/eBOSS_ELG/mocks/'
outputmul='/scratch2/scratchdirs/izamar/multipolos/output_mul/' #'/alicefs/marvam_g/bizb_a/eBOSS_ELG/multipolos/output_mul_mocks_rdzw/' #'/alicefs/marvam_g/bizb_a/eBOSS_ELG/multipolos/output_mul_mocks/'

inputCUTE_fileN='qpm_mock_anymask_ELG_NGC_spectweights_' #'qpm_mock_ELG_NGC_'
inputCUTE_fileS='qpm_mock_anymask_ELG_SGC_spectweights_' #'qpm_mock_ELG_SGC_'
outputCUTE_fileN='qpm_mock_anymask_ELG_NGC_spectweights_' #'qpm_mock_ELG_NGC_'
outputCUTE_fileS='qpm_mock_anymask_ELG_SGC_spectweights_' #'qpm_mock_ELG_SGC_'
randomCUTE_fileN='QPM_ELG_anymask_specweights_randoms200x_NGC.rdz' #'QPM_ELG_randoms50x_NGC.rdzw'
randomCUTE_fileS='QPM_ELG_anymask_specweights_randoms50x_SGC.rdz' #'QPM_ELG_randoms50x_SGC.rdzw'
outputmul_file='qpm_mock_anymask_ELG_spectweights_comb_' #'qpm_mock_ELG_comb_'  #cambiar nombre para NGC, SGC, comb


##post reconstruction
inputCUTE_recon='/global/cscratch1/sd/mmagana/EBOSS_ELG/v2.0/postrecon/'
#'/alicefs/marvam_g/bizb_a/eBOSS_ELG/mocks_recon/'
outputCUTE_recon='/scratch2/scratchdirs/izamar/output/'
#'/alicefs/marvam_g/bizb_a/eBOSS_ELG/output_mocks_recon/'
randomCUTE_recon='/global/cscratch1/sd/mmagana/EBOSS_ELG/v2.0/postrecon/'
#'/alicefs/marvam_g/bizb_a/eBOSS_ELG/mocks_recon/'
outputmul_recon='/scratch2/scratchdirs/izamar/multipolos/output_mul/'
#'/alicefs/marvam_g/bizb_a/eBOSS_ELG/multipolos/output_mul_mocks_recon'

inputCUTE_fileN_recon='qpm_mock_anymask_ELG_recon_specweightsNGC_' 
#'qpm_mock_ELG_reconNGC_' 
inputCUTE_fileS_recon='qpm_mock_anymask_ELG_recon_specweightsSGC_'
#'qpm_mock_ELG_reconSGC_' 
outputCUTE_fileN_recon='qpm_mock_anymask_ELG_recon_specweightsNGC_'
#'qpm_mock_ELG_reconNGC_' 
outputCUTE_fileS_recon='qpm_mock_anymask_ELG_recon_specweightsSGC_'
#'qpm_mock_ELG_reconSGC_' 
outputmul_file_recon='qpm_mock_anymask_ELG_recon_specweights_comb_' #cambiar nombre para NGC, SGC, comb
#'qpm_mock_ELG_recon_comb_' 

#para los SS de Landy-Szalay pos-recon
randomCUTE_reconN='QPM_ELG_anymask_randoms200x_recon_specweightsNGC_'
#'QPM_ELG_randoms20x_reconNGC_'
randomCUTE_reconS='QPM_ELG_anymask_randoms200x_recon_specweightsSGC_'
#'QPM_ELG_randoms20x_reconSGC_'
#el nombre de los archivos random cambia para cada mock
#QPM_ELG_randoms20x_reconSGC_0493_shuffle.rdzw


plotsdir='./'

def multipolos(NUM):

	if 0<=j<1000:

	 #para pre-recon

         norte=np.loadtxt(outputCUTE_N+outputCUTE_fileN+str(NUM+1).zfill(4)+'.xi', skiprows=0, unpack=True, usecols=[0,1,2,3,4,5,6])
         sur=np.loadtxt(outputCUTE_S+outputCUTE_fileS+str(NUM+1).zfill(4)+'.xi', skiprows=0, unpack=True, usecols=[0,1,2,3,4,5,6])

         datosN = np.loadtxt(inputCUTE+inputCUTE_fileN+str(NUM+1).zfill(4)+'.rdz', skiprows=0, unpack=True, usecols=[0,1,2,3])
         Nn = np.sum(datosN[3])

         datosS = np.loadtxt(inputCUTE+inputCUTE_fileS+str(NUM+1).zfill(4)+'.rdz', skiprows=0, unpack=True, usecols=[0,1,2,3])
         Ns = np.sum(datosS[3])

	 #Para post-recon
         norte_rec=np.loadtxt(outputCUTE_recon+outputCUTE_fileN_recon+str(NUM+1).zfill(4)+'.xi', skiprows=0, unpack=True, usecols=[0,1,2,3,4,5,6])
         sur_rec=np.loadtxt(outputCUTE_recon+outputCUTE_fileS_recon+str(NUM+1).zfill(4)+'.xi', skiprows=0, unpack=True, usecols=[0,1,2,3,4,5,6])

         datosN_rec = np.loadtxt(inputCUTE_recon+inputCUTE_fileN_recon+str(NUM+1).zfill(4)+'.rdz', skiprows=0, unpack=True, usecols=[0,1,2,3])
         Nn_rec = np.sum(datosN_rec[3])

         datosS_rec = np.loadtxt(inputCUTE_recon+inputCUTE_fileS_recon+str(NUM+1).zfill(4)+'.rdz', skiprows=0, unpack=True, usecols=[0,1,2,3])
         Ns_rec = np.sum(datosS_rec[3])

	 w_fkp_N_ran_rec = np.loadtxt(randomCUTE_recon+randomCUTE_reconN+str(NUM+1).zfill(4)+'.rdz',skiprows=0, unpack=True, usecols=[3])
	 Nrn_rec=sum(w_fkp_N_ran_rec) 	

	 w_fkp_S_ran_rec = np.loadtxt(randomCUTE_recon+randomCUTE_reconS+str(NUM+1).zfill(4)+'.rdz',skiprows=0, unpack=True, usecols=[3])
	 Nrs_rec=sum(w_fkp_S_ran_rec) 
	

	#w_fkp_N_ran = np.loadtxt(randomCUTE+randomCUTE_fileN,skiprows=0, unpack=True, usecols=[3])
	Nrn=9520209.64124  #sum(w_fkp_N_ran) #2627276.29951 #sum(w_fkp_N_ran)  #13729902.0 #sum(w_fkp_N_ran)
	#print Nrn
	#w_fkp_S_ran = np.loadtxt(randomCUTE+randomCUTE_fileS,skiprows=0, unpack=True, usecols=[3])
	Nrs=2672890.24968 #sum(w_fkp_S_ran)#11115841.8285 #611334.880698 #sum(w_fkp_S_ran)  #14164225.0 #sum(w_fkp_S_ran)
	#print Nrs
	
	bs=8

	##Para pre-recon
	
	
        DDn = norte[4]
        DDn = np.reshape(DDn,(200,100))
        DD1n = DDn[:200:8,:]
        for i in range(bs-1):
            DD1n=DD1n+DDn[i+1:200:8,:]
        DDn= DD1n/8.
	
        DDs = sur[4]
        DDs = np.reshape(DDs, (200,100))
        DD1s = DDs[:200:8,:]
        for i in range(bs-1):
            DD1s=DD1s+DDs[i+1:200:8,:]
        DDs= DD1s/8.
	
        DRn = norte[5]
        DRn =np.reshape(DRn, (200,100))
        DR1n = DRn[:200:8,:]
        for i in range(bs-1):
            DR1n=DR1n+DRn[1+i:200:8,:]
        DRn= DR1n/8.

	
        DRs = sur[5]
        DRs =np.reshape(DRs, (200,100))
        DR1s = DRs[:200:8,:]
        for i in range(bs-1):
            DR1s=DR1s+DRs[1+i:200:8,:]
        DRs= DR1s/8.
	

        RRn = norte[6]
        RRn = np.reshape(RRn, (200,100))
        RR1n = RRn[:200:8,:]
        for i in range(bs-1):
            RR1n=RR1n+RRn[1+i:200:8,:]
        RRn=RR1n/8.

	
        RRs = sur[6]
        RRs = np.reshape(RRs, (200,100))
        RR1s = RRs[:200:8,:]
        for i in range(bs-1):
            RR1s=RR1s+RRs[1+i:200:8,:]
        RRs=RR1s/8.
	

	#para utilizar solo el archivo norte/sur
	
        #DD =(DDn)/float(Nn*(Nn-1))
        #DR = (DRn)/float(Nn*(Nrn))
        #RR = (RRn)/float(Nrn*(Nrn-1))
	
        #DD =(DDs)/float(Ns*(Ns-1))
        #DR = (DRs)/float(Ns*(Nrs))
        #RR = (RRs)/float(Nrs*(Nrs-1))

	#Para utilizar norte y sur
        DD =(DDn + DDs)/float(Nn*(Nn-1)+Ns*(Ns-1))
        DR = (DRn + DRs)/float(Nn*(Nrn)+Ns*(Nrs))
        RR = (RRn + RRs)/float(Nrn*(Nrn-1)+Nrs*(Nrs-1))


        X =(DD-DR+RR)/(RR)

        r = norte[1][::100]
        mu = norte[0][0:100]
        r = (r[::8]-.5)+bs/2

        #rs = sur[1][::100]
        #mus = sur[0][0:100]
        #rs = (rs[::8]-.5)+bs/2
	


	##Para post-recon
	
        DDn_rec = norte_rec[4]
        DDn_rec = np.reshape(DDn_rec,(200,100))
        DD1n_rec = DDn_rec[:200:8,:]
        for i in range(bs-1):
            DD1n_rec=DD1n_rec+DDn_rec[i+1:200:8,:]
        DDn_rec= DD1n_rec/8.

	
        DDs_rec = sur_rec[4]
        DDs_rec = np.reshape(DDs_rec, (200,100))
        DD1s_rec = DDs_rec[:200:8,:]
        for i in range(bs-1):
            DD1s_rec=DD1s_rec+DDs_rec[i+1:200:8,:]
        DDs_rec= DD1s_rec/8.


        DSn = norte_rec[5]
        DSn =np.reshape(DSn, (200,100))
        DS1n = DSn[:200:8,:]
        for i in range(bs-1):
            DS1n=DS1n+DSn[1+i:200:8,:]
        DSn= DS1n/8.

	
        DSs = sur_rec[5]
        DSs =np.reshape(DSs, (200,100))
        DS1s = DSs[:200:8,:]
        for i in range(bs-1):
            DS1s=DS1s+DSs[1+i:200:8,:]
        DSs= DS1s/8.
	

        SSn = norte_rec[6]
        SSn = np.reshape(SSn, (200,100))
        SS1n = SSn[:200:8,:]
        for i in range(bs-1):
            SS1n=SS1n+SSn[1+i:200:8,:]
        SSn=SS1n/8.

	
        SSs = sur_rec[6]
        SSs = np.reshape(SSs, (200,100))
        SS1s = SSs[:200:8,:]
        for i in range(bs-1):
            SS1s=SS1s+SSs[1+i:200:8,:]
        SSs=SS1s/8.
	

	#para utilizar solo archivo norte
        #DD_rec =(DDn_rec)/float(Nn_rec*(Nn_rec-1))
        #DS = (DSn)/float(Nn_rec*(Nrn_rec))
        #SS = (SSn)/float(Nrn_rec*(Nrn_rec-1))


        #para utilizar solo archivo sur
        #DD_rec =(DDs_rec)/float(Ns_rec*(Ns_rec-1))
        #DS = (DSs)/float(Ns_rec*(Nrs_rec))
        #SS = (SSs)/float(Nrs_rec*(Nrs_rec-1))


	#para utilizar archivos norte y sur
        DD_rec =(DDn_rec + DDs_rec)/float(Nn_rec*(Nn_rec-1)+Ns_rec*(Ns_rec-1))
        DS = (DSn + DSs)/float(Nn_rec*(Nrn_rec)+Ns_rec*(Nrs_rec))
        SS = (SSn + SSs)/float(Nrn_rec*(Nrn_rec-1)+Nrs_rec*(Nrs_rec-1))


        X_rec =(DD_rec-DS+SS)/(RR)

        #r_rec = norte_rec[1][::100]
        #mu_rec = norte_rec[0][0:100]
        #r_rec = (r_rec[::8]-.5)+bs/2

        #rs_rec = sur_rec[1][::100]
        #mus_rec = sur_rec[0][0:100]
        #rs_rec = (rs_rec[::8]-.5)+bs/2

	#para los archivos combinados(puedo tomar norte o sur)
        r_rec = norte_rec[1][::100]
        mu_rec = norte_rec[0][0:100]
        r_rec = (r_rec[::8]-.5)+bs/2
	
	#para los archivos norte
        X=X.reshape(len(r),len(mu))
        X_rec=X_rec.reshape(len(r_rec),len(mu_rec))

	#para los archivos sur
        #X=X.reshape(len(rs),len(mus))
        #X_rec=X_rec.reshape(len(rs_rec),len(mus_rec))

	#para los archivos norte
        mono = np.zeros(len(r))
        quad = np.zeros(len(r))

        mono_rec = np.zeros(len(r_rec))
        quad_rec = np.zeros(len(r_rec))


        #para los archivos sur
        #mono = np.zeros(len(rs))
        #quad = np.zeros(len(rs))

        #mono_rec = np.zeros(len(rs_rec))
        #quad_rec = np.zeros(len(rs_rec))

	#### recordar cambiar los rangos para el tipo de archivo correspondiente
        for ii in range(len(r)):
            for jj in range(len(mu)):
                mono[ii] += X[ii,jj]/100.
                quad[ii] += (5/2.)*X[ii,jj]*(3.*mu[jj]*mu[jj]-1.)/100.
                mono_rec[ii] += X_rec[ii,jj]/100.
                quad_rec[ii] += (5/2.)*X_rec[ii,jj]*(3.*mu_rec[jj]*mu_rec[jj]-1.)/100.
        return r,mono,quad,r_rec,mono_rec,quad_rec


NUM=range(448,500)#range(float(inputp),float(inputq))   #,str(inputq))


for j in NUM:

        multipoles=multipolos(j)
        r = multipoles[0]
        mono = multipoles[1]
        quad = multipoles[2]
        r_rec = multipoles[3]
        mono_rec = multipoles[4]
        quad_rec = multipoles[5]


	salida=open(outputmul+outputmul_file+str(j+1).zfill(4)+'.mul', 'w')

        for i in range(len(r)):
           linea='{} {} {} \n'.format(str(r[i]),str(mono[i]),str(quad[i]))
           salida.writelines(linea)

        salida.close()

	salida_rec=open(outputmul_recon+outputmul_file_recon+str(j+1).zfill(4)+'.mul', 'w')

        for i in range(len(r_rec)):
           linea='{} {} {} \n'.format(str(r_rec[i]),str(mono_rec[i]),str(quad_rec[i]))
           salida_rec.writelines(linea)

        salida_rec.close()

	"""
        plt.plot(multipoles[0],multipoles[0]**2*multipoles[1],'k')
        #plt.plot(monopole[0],monopole[0]**2*monopole[1],'k')
                                                                                       
        plt.title('ELGs pre-recon Monopoles derived from $\\xi(\mu,r)$')
        plt.xlim(0,200)
        plt.ylim(-200,200)
        plt.xlabel('$r$')
        plt.ylabel('$\\xi_0(r)r^2$')
	plt.show()
        #plt.savefig(plotsdir+'ELGs_monopoles_'+str(j)+str(i)+'.pdf')
	#plt.savefig(plotsdir+'ELGs_pre-rec_monopoles.pdf')

        plt.plot(multipoles[0],multipoles[0]**2*multipoles[2],'k')
        plt.title('ELGs pre-recon Quadrupoles derived from $\\xi(\mu,r)$')
        plt.xlim(0,200)
        plt.ylim(-200,200)
        plt.xlabel('$r$')
        plt.ylabel('$\\xi_2(r)r^2$')
	plt.show()
	#plt.savefig(plotsdir+'ELGs_pre-rec_quadrupoles.pdf')

 	###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        plt.plot(multipoles[3],multipoles[3]**2*multipoles[4],'k')                                
        plt.title('ELGs post-recon Monopoles derived from $\\xi(\mu,r)$')
        plt.xlim(0,200)
        plt.ylim(-200,200)
        plt.xlabel('$r$')
        plt.ylabel('$\\xi_0(r)r^2$')
	plt.show()
        #plt.savefig(plotsdir+'ELGs_monopoles_'+str(j)+str(i)+'.pdf')
	#plt.savefig(plotsdir+'ELGs_post-rec_monopoles.pdf')

        plt.plot(multipoles[3],multipoles[3]**2*multipoles[5],'k')
        plt.title('ELGs post-recon Quadrupoles derived from $\\xi(\mu,r)$')
        plt.xlim(0,200)
        plt.ylim(-200,200)
        plt.xlabel('$r$')
        plt.ylabel('$\\xi_2(r)r^2$')
	plt.show()
	#plt.savefig(plotsdir+'ELGs_post-rec_quadrupoles.pdf')
	"""
#plt.savefig(plotsdir+'ELGs_quadrupoles.pdf')

