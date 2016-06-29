import numpy as np
from glob import glob
import os
import h5py
#....
#datadir = "/reg/d/ana01/temp/davidsch/ImgMLearnFull"
#h5files = glob(os.path.join(datadir, "amo86815_mlearn-r070*.h5"))

datadir = "/reg/d/ana01/temp/davidsch/ImgMLearnFull"
h5files = glob(os.path.join(datadir, "amo86815_pred-r072*.h5"))
avgiter = 0
waveformsiter= np.zeros((5))
itertot = 0
avgwaveform = np.zeros((1,1000))
waveforms = np.zeros((5,1000))
tot = np.zeros((1,1000))
TrainData3 = np.load("RegressionFeatures.npy")
Y3 = np.load("RegresionY.npy")
TrainData2 = np.load("RegressionFeatures2.npy")
Y2 = np.load("RegresionY2.npy")
TrainData1 = np.load("RegressionFeatures1.npy")
Y1 = np.load("RegresionY1.npy")
from sklearn import linear_model
clf31 = linear_model.LinearRegression()
clf32 = linear_model.LinearRegression()
clf2 = linear_model.LinearRegression()
clf1 = linear_model.LinearRegression()
clf31.fit(TrainData3,Y3[:,1])
clf32.fit(TrainData3,Y3[:,3])
clf2.fit(TrainData2,Y2[:,3])
clf1.fit(TrainData1,Y1[:,1])
stor1=[]
stor2=[]
stor3=[]

l = 4.5
u = 4.6

for idx, h5file in enumerate(h5files):
    h5 = h5py.File(h5file,'r')
    for k in range(len(h5['predict_enPeaks'])):
        temp = h5['predict_enPeaks'][k]
        if temp ==1:
            #t = h5['bld.ebeam.ebeamL3Energy'][k]
            #L3Values1.append(t)
            a = h5['predict_enPeaks_MrgL02_relu_act'][k]
            b = h5['predict_enPeaks_MrgL05_relu_act'][k]
            c = h5['predict_enPeaks_MrgL08_relu_act'][k]
            temp1 = np.concatenate((a,b,c),axis =0)
            q = clf1.predict(temp1)
            stor1.append(q[0])
            if( q[0] > l and q[0] < u):
                avgwaveform = avgwaveform + h5['acq.waveforms'][k][5,:]
                avgiter = avgiter + 1
        if temp ==3:
            #t = h5['bld.ebeam.ebeamL3Energy'][k]
            a = h5['predict_enPeaks_MrgL02_relu_act'][k]
            b = h5['predict_enPeaks_MrgL05_relu_act'][k]
            c = h5['predict_enPeaks_MrgL08_relu_act'][k]
            temp1 = np.concatenate((a,b,c),axis =0)
            q = clf31.predict(temp1)
            q2 = clf32.predict(temp1)
            stor3.append(q[0])
            stor2.append(q2[0])
            
            if(q[0] >l and q[0] < u and q2[0] >2.9  and q2[0]<3.0):
                waveforms[0,:] = waveforms[0,:] + h5['acq.waveforms'][k][5,:]
                tot = tot + h5['acq.waveforms'][k][5,:]
                itertot = itertot + 1
                waveformsiter[0] +=1
            if(q[0] >l and q[0] < u and q2[0] >3.0  and q2[0]<3.1):
                waveforms[1,:] = waveforms[1,:] + h5['acq.waveforms'][k][5,:]
                tot = tot + h5['acq.waveforms'][k][5,:]
                itertot = itertot + 1
                waveformsiter[1] +=1
            if(q[0] >l and q[0] < u and q2[0] >3.1  and q2[0]<3.2):
                waveforms[2,:] = waveforms[2,:] + h5['acq.waveforms'][k][5,:]
                tot = tot + h5['acq.waveforms'][k][5,:]
                itertot = itertot + 1
                waveformsiter[2] +=1
            if(q[0] >l and q[0] < u and q2[0] >3.2  and q2[0]<3.3):
                waveforms[3,:] = waveforms[3,:] + h5['acq.waveforms'][k][5,:]
                tot = tot + h5['acq.waveforms'][k][5,:]
                itertot = itertot + 1
                waveformsiter[3] +=1
            if(q[0] >l and q[0] < u and q2[0] >3.3  and q2[0]<3.4):
                waveforms[4,:] = waveforms[4,:] + h5['acq.waveforms'][k][5,:]
                tot = tot + h5['acq.waveforms'][k][5,:]
                itertot = itertot + 1
                waveformsiter[4] +=1
 
           




avgwaveform = avgwaveform/avgiter
tot = tot/itertot
for k in range(5):
    waveforms[k,:] = waveforms[k,:]/waveformsiter[k]
    
tot = tot[0,:] / np.max(tot[0,:])
avgwaveform = avgwaveform[0,:] / np.max(avgwaveform[0,:])
for k in range(5):
    waveforms[k,:] = waveforms[k,:] / np.max(waveforms[k,:])
    
import matplotlib.pyplot as plt

plt.plot(avgwaveform[300:700]+ 0)
plt.plot(tot[300:700] + 1)
plt.plot(waveforms[0,300:700]+2)
plt.plot(waveforms[1,300:700]+3)
plt.plot(waveforms[2,300:700]+4)
plt.plot(waveforms[3,300:700]+5)
plt.plot(waveforms[4,300:700]+6)

