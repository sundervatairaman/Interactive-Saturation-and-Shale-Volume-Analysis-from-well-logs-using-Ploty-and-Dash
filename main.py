import threading
from dash import Dash, html, dcc, Input, Output
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import os
import asyncio
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

import lasio

from bokeh.models.formatters import PrintfTickFormatter
from bokeh.models.widgets import Button

from scipy.optimize import minimize
import ipywidgets as widgets

#%matplotlib inline

import panel as pn

#Qt4Agg

# Suggested by Hoxbro at Panel
#plt.switch_backend("agg")
#plt.switch_backend("Qt5Agg")


pn.extension()





# Initialize Dash app
dash_app = Dash(__name__)

# Define initial values for sliders
# GR endpoint parameters for shale calculations
gr_clean, gr_shale      = 45  , 120                             # Shale Parmaetrs for GR
  
  # SP endpoint parameters for shale calculations
sp_clean, sp_shale      = -50  , 0                              # Shale Parameters for SP
  
  # Endpoint Parameters for Neutron-Density Shale Triangle
neut_shale, den_shale   = 0.32 , 2.65                           # Shale Parmaters for Neutron-Density
neut_matrix, den_matrix = -0.04 , 2.65                          # Matrix Parmameters for Neutron-Density
neut_fl, den_fl         =  1.0 , 1.1                            # Fluid Parmameters for Neutron-Density
  
  # Fluid Parameters for MRIL tool
mphi_shale              = 0.0                                   # MRIL Parameters
  
  # Fluid Parameters for Sonic tool
dt_matrix, dt_fluid, dt_shale, cp, alpha = 55.5,188.0,90.0,1,5/8 # Sonic Parameters
  
  # Rt Shale parameters
rt_shale = 6
rt_clean = 170
  
m_cem, n_sat, mslope, Rw = 1.9, 2.0, 1.5, 0.15  # Fluid Parameters





alias = {
            'sonic': ['none', 'DTC', 'DT24', 'DTCO', 'DT', 'AC', 'AAC', 'DTHM'],
            'ssonic': ['none', 'DTSM', 'DTS'],
            'gr': ['none', 'GR', 'GRD', 'CGR', 'GRR', 'GRCFM'],
            'resdeep': ['none', 'HDRS', 'LLD', 'M2RX', 'MLR4C', 'RD', 'RT90', 'RLA1', 'RDEP', 'RLLD', 'RILD', 'ILD', 'RT_HRLT', 'RACELM'],
            'resshal': ['none', 'LLS', 'HMRS', 'M2R1', 'RS', 'RFOC', 'ILM', 'RSFL', 'RMED', 'RACEHM', 'M2R6'],
            'density': ['none', 'ZDEN', 'RHOB', 'RHOZ', 'RHO', 'DEN', 'RHO8', 'BDCFM'],
            'neutron': ['none', 'CNCF', 'NPHI', 'NEU'],
            'depth': ['none','DEPTH', 'DEPT', 'MD', 'TVD', 'MPHI'],  # Depth aliases
            'pe': ['none','PE', 'PEF', 'PEFZ'],  # Photoelectric factor aliases
            'caliper': ['none','CALI', 'CAL', 'CALI2', 'CALX'],  # Caliper log aliases
            'bs': ['none','BS', 'BIT', 'BITSIZE', 'BDT','BS1'],  # Bit size aliases
            'vpvs': ['none','VPVS', 'VP_VS', 'VPVS_RATIO', 'VPVS_R'],  # Vp/Vs ratio aliases
            'rxo': ['none','RXO', 'RMLL', 'Rxo', 'RTXO', 'RSXO', 'RMSFL', 'MSFL', 'RXXO', 'RX', 'M2R6','M2R2'],
            'sp': ['none','SP', 'SSP', 'SPONT', 'SPONPOT', 'SPOT', 'SPT', 'SP_CURVE']
        }


import lasio

class LASFileHandler:
    def __init__(self):
        self.laspath = None

    def select_las_file(self):
        app = QApplication(sys.argv)  # Create an application instance
        file_dialog = QFileDialog()   # Create a QFileDialog instance
        file_dialog.setNameFilter("LAS Files (*.las)")  # Set filter to show only .las files
        file_dialog.setWindowTitle("Select a LAS file")  # Dialog title
        if file_dialog.exec_():       # Open the dialog
            self.laspath = file_dialog.selectedFiles()[0]  # Retrieve the selected file path
            print(f"File selected: {self.laspath}")
        else:
            print("File selection was canceled.")
        app.exit()  # Cleanly exit the application instance

    def read_las_file(self):
        if self.laspath:
            try:
                las = lasio.read(self.laspath)
                print(f"Loaded LAS file: {self.laspath}")
                # Example: printing well information
                for item in las.well:
                    print(f"{item.descr} ({item.mnemonic}): {item.value}")
                for curve in las.curves:
                    print(f"Curve: {curve.mnemonic}, Units: {curve.unit}, Description: {curve.descr}")
                return las
            except Exception as e:
                print("Error reading LAS file:", e)
        else:
            print("No file selected to read.")

# Example usage
handler = LASFileHandler()

# Select the LAS file using a file dialog
handler.select_las_file()

# After selecting the file, read it
las=handler.read_las_file()


#las = lasio.read("WADU_72.las")
#las = lasio.read("./data/NMR_Well.las")
#las = lasio.read("./data/ws_petrogg.las")


for item in las.well:
    print(f"{item.descr} ({item.mnemonic}): {item.value}")

for curve in las.curves:
    print(curve.mnemonic)

header = [curve.mnemonic for curve in las.curves]


for count, curve in enumerate(las.curves):
    print(f"Curve: {curve.mnemonic}, Units: {curve.unit}, Description: {curve.descr}")
print(f"There are a total of: {count+1} curves present within this file")

data = las.df()    #store las file in df variable as pandas dataframe


data.describe()


data.info()

alias['gr'] = [elem for elem in header if elem in set(alias['gr'])]
alias['sonic'] = [elem for elem in header if elem in set(alias['sonic'])]
alias['ssonic'] = [elem for elem in header if elem in set(alias['ssonic'])]
alias['resdeep'] = [elem for elem in header if elem in set(alias['resdeep'])]
alias['resshal'] = [elem for elem in header if elem in set(alias['resshal'])]
alias['density'] = [elem for elem in header if elem in set(alias['density'])]
alias['neutron'] = [elem for elem in header if elem in set(alias['neutron'])]
    
alias['depth'] = [elem for elem in header if elem in set(alias['depth'])]
alias['pe'] = [elem for elem in header if elem in set(alias['pe'])]
alias['caliper'] = [elem for elem in header if elem in set(alias['caliper'])]
alias['bs'] = [elem for elem in header if elem in set(alias['bs'])]

alias['vpvs'] = [elem for elem in header if elem in set(alias['vpvs'])]
alias['rxo'] = [elem for elem in header if elem in set(alias['rxo'])]

alias['sp'] = [elem for elem in header if elem in set(alias['sp'])]

print('test')


    #data=data.rename(columns=({'DEPTH':'DEPT'}))
    #data['DEPT']=data.index
    #data['DEPTH']=data[alias['depth'][0]].values
print('test5')
data=data.rename(columns=({'DEPTH':'DEPT'}))
    #data['DEPT']=data['MPHI']

data['DEPT']=data.index
#data['Rxo'] = data[alias['rxo'][0]].values
data['BS'] = data[alias['bs'][0]].values
data['CALI']= data[alias['caliper'][0]].values

if not alias['rxo']:
        data['Rxo'] = 0
    
else:
        data['Rxo'] = data[alias['rxo'][0]].values




if not alias['sonic']:
        data['DT'] = 0
    
else:
        data['DT'] = data[alias['sonic'][0]].values

print('test6')

data['NPHI'] = data[alias['neutron'][0]].values

if data['NPHI'].max() > 30:
    data['NPHI'] = data['NPHI'] / 100


data['DCAL'] = data['CALI'] - data['BS']
    
if not alias['sp']:
        data['SP'] = 0
    
else:

        data['SP']  = data[alias['sp'][0]].values

    


data['GR']= data[alias['gr'][0]].values

data['ILD'] = data[alias['resdeep'][0]].values
print('test1')
data['ILM'] = data[alias['resshal'][0]].values

data['RHOB'] = data[alias['density'][0]].values
print('test2')

if not alias['ssonic']:
        data['DTS'] = 0
    
else:

        data['DTS'] = data[alias['ssonic'][0]].values

print('test')
data=data.rename(columns=({'DEPTH':'DEPT'}))

data['VPVS'] = 0
data['YMOD'] = 0
data['MPHI'] = 0

    #data['DEPT']=data.index
    #data['MPHI'] = data['MPHI']/100
    #data['MBVI'] = data['MBVI']/100




    #data['LL8'] = data['RMLL']
    #data['DT'] = data['GR']
    #data['PHIX'] = data['NPHI']/100
    #data['MPHI'] = data['MPHI']/100
    #data['MBVI'] = data['MBVI']/100
    #data['NPHI'] = data['NPHI']/100
print(data)

#data=data.rename(columns=({'DEPTH':'DEPT'}))
#data['DEPT']=data.index
#data['Rxo'] = data['RMLL']
#data['BS'] = 10
#data['DTC'] = data['DT']
#data['NPHI'] = data['NPHI'] / 100
#data['DCAL'] = data['CALI'] - data['BS']
#data['DEPT']=data.index
#data['MPHI'] = data['MPHI']/100
#data['MBVI'] = data['MBVI']/100




#data['LL8'] = data['RMLL']
#data['DT'] = data['GR']
#data['PHIX'] = data['NPHI']/100
#data['MPHI'] = data['MPHI']/100
#data['MBVI'] = data['MBVI']/100
#data['NPHI'] = data['NPHI']/100

data.describe()


"""
===============================================================================
                    Depths over most of well interval
===============================================================================

===============================================================================
 a Main ZONE for analysis will be selected from the entire log. Program displays
 the logs again within the choosen interval with triple_combo_plot
 function.
 
 === Select zone of analysis: top and depth
 
 by setting the ``top_depth'' and ``bottom_depth'' variables
===============================================================================
"""
min_depth1 = int(data['DEPT'].min())
max_depth1 = int(data['DEPT'].max())

logs = data[(data.DEPT >= min_depth1) & (data.DEPT <= max_depth1)]



#vshgr
  #---the setup below is correction = None or linear GR as recommended by Heslep
def vshgr(gr_log, gr_clean, gr_shale, correction=None):
  
      igr=(gr_log-gr_clean)/(gr_shale-gr_clean)      #Linear Gamma Ray
      vshgr_larionov_young=0.083*(2**(3.7*igr)-1)   #Larionov (1969) - Tertiary rocks
      vshgr_larionov_old=0.33*(2**(2*igr)-1)        #Larionov (1969) - Older rocks
      vshgr_clavier=1.7-(3.38-(igr+0.7)**2)**0.5    #Clavier (1971)
      vshgr_steiber=0.5*igr/(1.5-igr)               #Steiber (1969) - Tertiary rocks
  
      if correction == "young":
          vshgr=vshgr_larionov_young
      elif correction == "older":
          vshgr=vshgr_larionov_old
      elif correction=="clavier":
          vshgr=vshgr_clavier
      elif correction=="steiber":
          vshgr=vshgr_steiber
      else:
          vshgr=igr
      return vshgr
  
def vshgr2(gr_log, gr_clean, gr_shale):
      (gr_log - gr_clean)/(gr_shale - gr_clean)      #Linear Gamma Ray
      return vshgr
  
  #vshsp
def vshsp(sp_log, sp_clean, sp_shale):
      vshsp=(sp_log-sp_clean)/(sp_shale-sp_clean)
      return vshsp
  
  #dt_matrix, dt_fluid, dt_shale
  #vsh dt
def vshdt(dt_log, dt_matrix, dt_shale):
      vshdt=(dt_log - dt_matrix)/(dt_shale-dt_matrix)
      return vshdt
  
  
  #vshrt
def vshrt(rt_log, rt_clean,rt_shale):
      vrt=(rt_shale/rt_log)*(rt_clean-rt_log)/(rt_clean-rt_shale)
      if (rt_log > 2* rt_shale):
          vshrt = 0.5 * (2 * vrt)** (0.67*(vrt+1)) 
      else:
          vshrt = vrt
      return vshrt
  
  
  #vshnd
def vshnd(neut_log,den_log,neut_matrix,den_matrix,neut_fl,den_fl,neut_shale,den_shale):
      term1 = (den_fl-den_matrix)*(neut_log-neut_matrix)-(den_log-den_matrix)*(neut_fl-neut_matrix)
      term2 =(den_fl-den_matrix)*(neut_shale-neut_matrix)-(den_shale-den_matrix)*(neut_fl-neut_matrix)
      vshnd=term1/term2
      return vshnd
  
  # vsh Dt vs. bulk density
def vshdtden(dt_log,den_log,  dt_matrix,den_matrix,  dt_fluid,den_fl,  dt_shale,den_shale):
      term1 = (den_fl - den_matrix)*(dt_log-dt_matrix)   - (den_log - den_matrix)*(dt_fluid - dt_matrix)
      term2 = (den_fl - den_matrix)*(dt_shale-dt_matrix) - (den_shale-den_matrix)*(dt_fluid - dt_matrix)
      vshdtden=term1/term2
      return vshdtden
  
  
  
  #vsh_Neutron_MPHI
def vshnmphi(nphi, mphi , nphi_sh, mphi_sh):
      #--------------------------------------------------
      #
      #    COMPUTE VSH FROM MPHI-NPHI Shale TRIANGLE 
      #
      #--------------------------------------------------    
      #mphi_shale = 0.0
      phi = (mphi * nphi_sh - nphi * mphi_sh)/(nphi_sh - mphi_sh)
      vshnmphi = (nphi - phi)/(nphi_sh)
      return vshnmphi
   
  # ======= This was the original from Mahai and I do not use clay ==============
  # neut_matrix, den_matrixtrix = 15, 2.6 #Define clean sand line 
  # neut_fl, den_fl = 40, 2 
  # =============================================================================
 
# Input parameters for Top and Bottom depths 
##################logs=data[(data.DEPT >= top_depth) & (data.DEPT <= bottom_depth)]
  
  # calculate the vsh functions, by looping with pandas series values through vsh functions defined above
  #  without looping - the function will throw an error
  
  #initialize various Vsh
  #vshgr_temp,vshnd_temp, vshrt_temp, vshsp_temp , vshnmphi_temp =[],[],[],[], []
vshgr_temp,vshnd_temp, vshrt_temp, vshsp_temp , vshdt_temp, vshrt_temp,  vshnmphi_temp, vshdtden_temp =[],[],[],[], [], [], [], []
  
  
  # ===== this is an example of for a,b in zip(alist,blist): ====================
  # alist = ['a1', 'a2', 'a3']
  # blist = ['b1', 'b2', 'b3']
  # 
  # for a, b in zip(alist, blist):
  #     print a, b
  # 
  # =============================================================================
  # =============================================================================
  # This is key for the input of log data to be used in Vsh calculations
  # =============================================================================
  #for (i,j,k,l,m,n) in zip(logs.GR,logs.NPHI,logs.RHOB,logs.ILD,logs.SP,logs.MPHI):
  #    vshgr_temp.append(vshgr(i, gr_clean, gr_shale))
  #    vshnd_temp.append(vshnd(j,k,neut_matrix,den_matrix,neut_fl,den_fl,neut_shale,den_shale))    
  #    vshsp_temp.append(vshsp(m, sp_clean, sp_shale))
  #    vshnmphi_temp.append(vshnmphi(j, n, neut_shale, 0))
  #
  # =============================================================================
  # This is key for the input of log data to be used in Vsh calculations
  # =============================================================================
  #                           i       j         k         l      m        n
for (i,j,k,l,m,n,o) in zip(logs.GR,logs.NPHI,logs.RHOB,logs.ILD,logs.SP,logs.DT,logs.MPHI):
      vshgr_temp.append(vshgr(i, gr_clean, gr_shale))
      vshnd_temp.append(vshnd(j,k,neut_matrix,den_matrix,neut_fl,den_fl,neut_shale,den_shale))    
      vshsp_temp.append(vshsp(m, sp_clean, sp_shale))
      vshdt_temp.append(vshdt(n, dt_matrix, dt_shale) )   
      vshrt_temp.append(vshrt(j, rt_clean, rt_shale) )   
      vshdtden_temp.append(vshdtden(n,k,  dt_matrix,den_matrix,  dt_fluid,den_fl,  dt_shale,den_shale) )
      vshnmphi_temp.append(vshnmphi(j, o, neut_shale, 0))
  # =============================================================================
      
      
  # =============================================================================
  # This is the input of log data used in Vsh calculations
  # =============================================================================
  #======== test this as it prints GR, NPHI, RHOB, ILD, SP  =====================
  #    print(i,j,k,l,m) #where i=GR, j=NPHI, k=RHOB, l=ILD and m=SP
  #==============================================================================
      
logs.is_copy = False # without will throw an exception
  
logs['vshgr']=vshgr_temp
logs['vshnd']=vshnd_temp
logs['vshsp']=vshsp_temp
logs['vshdt']=vshdt_temp
logs['vshrt']=vshrt_temp
logs['vshdtden']=vshdtden_temp
logs['vshnmphi']=vshnmphi_temp
  
print( logs['vshgr']) 
del vshgr_temp, vshnd_temp, vshsp_temp, vshdt_temp, vshrt_temp, vshnmphi_temp         #remove the arrays to free-up memory

logs.head()
  
  
checkbox_group = pn.widgets.CheckBoxGroup(
      name='Shale Indicators', value=['vshgr', 'vshsp','vshnmphi','vshnd'], options = ["vshgr", "vshsp", "vshnd", "vshdt", "vshdtden", "vshnmphi"] ,
      inline=True)
  
checkbox_group.value=["vshgr",  "vshnd"]
  
  
print(checkbox_group.value)
vsh_array = logs[checkbox_group.value].to_numpy()
print(vsh_array)
  
  # Convert specific columns 
  #array = logs[['vshgr', 'vshsp','vshnd','vshnmphi']].to_numpy()
  #array.shape   
  
medianvsh = []
  
for i in range(0,len(logs.vshgr),1):
      
      #medianvsh.append(np.median(array[i]))
      medianvsh.append(np.median(vsh_array[i]))
  
      
logs['vshmedian'] = medianvsh
  
print(logs['vshmedian'])
  
  
import itertools
import statistics
  
def hodges_lehmann(data):
      pairs = list(itertools.combinations(data, 2))
      averages = [(x + y) / 2 for x, y in pairs]
      hodges_lehmann_median = statistics.median(averages)
      return hodges_lehmann_median
  
  
  # Convert specific columns 
  #array = logs[['vshgr', 'vshsp','vshnd','vshnmphi']].to_numpy()
  
vsh_hl = []
  
for i in range(0,len(logs),1):
      
      #vsh_hl.append(hodges_lehmann(array[i]))
      vsh_hl.append(hodges_lehmann(vsh_array[i]))
  
      
logs['vsh_hl'] = vsh_hl
print(logs.vsh_hl)






  
# calculate the vsh functions, by looping with pandas series values through vsh functions defined above
 #==============================================================================
T   = 170.0       # Reservoir temperature in DegF
  
TC  = (T-32)/1.8  # Temp DegC
  
Rw75  = ((T+6.77)*Rw)/(75+6.77)
  
  # Salinity in KPPM
SAL   = (10**((3.562-math.log10(Rw75-0.0123))/0.955))/1000
  
B     = math.exp(7.3-28.242/math.log(T)-0.2266*math.log(Rw)) 
  
Bdacy = (1-0.83*math.exp(-math.exp(-2.38+(42.17/TC))/Rw))*(-3.16+1.59*math.log(TC))**2 #SCA Paper SCA2006-29 (3)
  
print('Res temp =', T, 'Rw at Res Temp =',Rw, 'Rw@75 =', Rw75, 'B =',B, 'Bdacy =',Bdacy,'SAL =', SAL)      
  
  
rdbuttons = pn.widgets.RadioBoxGroup(
      options = ['CNL_1pt0','CNL_1pt1','TNPH_1pt0', 'TNPH_1pt19'],
      value='CNL_1pt1',
      #layout={'widdt':'max-content'},
      name = 'Neutron Log',
      disabled=False
)
  
  
print()
print('Select the Neutron log and Salinity Chartbook to use for SLB tools only at this point:')
print()
  
rdbuttons
  
  
chart=rdbuttons.value
print('Neutron-Density Chart =', chart)
print()
  
  
  
#select the proper Neutron-Denisity Chartbook file
''' 
    Schlumberger CNL Neutron Log at different Fluid Densities
'''
#file = r'./data/CNL_1pt0.xlsx'
#file = r'./data/CNL_1pt1.xlsx'
''' 
      Schlumberger TNPH Neutron Log at different Fluid Densities
'''
#file = r'./data/TNPH_1pt0.xlsx'
#file = r'./data/TNPH_1pt19.xlsx'
  
if chart == 'CNL_1pt0':
      file = r'./data/CNL_1pt0.xlsx'
elif chart == 'CNL_1pt1':
      file = r'./data/CNL_1pt1.xlsx'
elif chart == 'TNPH_1pt0':
      file = r'./data/TNPH_1pt0.xlsx'
else:
      file = r'./data/TNPH_1pt19.xlsx'
df_chart = pd.read_excel(file,index_col=False)
  
CNL_chart  = df_chart['Neutron']
RHOB_chart = df_chart['RHOB']
Rho_Matrix_chart  = df_chart['Rho_Matrix']
Porosity_chart = df_chart['Porosity']   
  







def PHIT_Knn(CNL,RHOB):
      """
      # =============================================================================
      # # ===========================================================================
      # # #--------------------------------------------------------------------------
      # ##
      # ##            This is the beginnin of KNN estimating ND xplt Porosity 
      # ##
      # # #--------------------------------------------------------------------------
      # # ===========================================================================
      # =============================================================================
      """  
  
  
  
  
      #deptharray = []
      porarray   = []; #make list of 0 length
      RHOMAA_array = []
      Porosity_array = []
      rhoarray = []
  
  
      #log Data
      for k in range(0,len(logs) ,1):  
  
              cnl_norm  = (CNL[k]-(-0.05))/(0.6-(-0.05))
              rhob_norm = (RHOB[k]-1.9)/(3-1.9)
  
  
  
              dist_inv    = []
              dist_cnl    = []
              dist_rhob    = []
              inv_dist_array = []
              Por_weight = []
              Rhomatrix_weight =[]
              CNL_norm = []
              RHOB_norm = []
  
              dist_inv_total = 0
              Por_total     = 0
  
  
  
              #this is the chartbook_reference_data being used 
              for i in range(0,len(df_chart),1):
  
                      CNL_norm.append((CNL_chart[i] - (-0.05)) / (0.6 - (-0.05)))
                      RHOB_norm.append((RHOB_chart[i] - 1.9) / (3.0 - 1.9))
  
                      #Euclidian Distance
                      dist_cnl.append((abs(cnl_norm   - CNL_norm[i])))
                      dist_rhob.append( abs(rhob_norm - RHOB_norm[i]))
  
                      if math.sqrt(dist_cnl[i]**2 + dist_rhob[i]**2) > 0:
                          dist_inv.append( 1  /  math.sqrt( dist_cnl[i]**2 + dist_rhob[i]**2)  )
                      else:
                          dist_inv.append( 1  /  math.sqrt( 0.0001 + dist_cnl[i]**2 + dist_rhob[i]**2)  )
  
  
                      #calculalte weights
                      Por_weight      .append(dist_inv[i] * Porosity_chart[i])
                      Rhomatrix_weight.append(dist_inv[i] * Rho_Matrix_chart[i])
  
  
  
                      inv_dist_array.append(dist_inv[i]);  # add items
  
              # =============================================================================
              ###                    KNN Array
              # # ===========================================================================
              # # #--------------------------------------------------------------------------
                      distance_knn_array = [dist_inv, Por_weight, Rhomatrix_weight]
              #        distance_knn_array = [Permeability, Porosity, G1, PD1, BV1, G2, PD2, BV2]
              # # #--------------------------------------------------------------------------
              # # ===========================================================================
              # =============================================================================
              xnorm=np.array(CNL_norm)
              ynorm=np.array(RHOB_norm)
  
  
              #knn_array = np.transpose array
              knn_array = np.transpose(distance_knn_array)
              #print(knn_array)
  
              #Sort array from large to low by column 0 which is dist_inv 
              #xknn=np.array(knn_array)
  
              #matsor x[x[:,column].argsort()[::-1]] and -1 us reverse order
              mat_sort = knn_array[knn_array[:,0].argsort()[::-1]] #firt column reverse sort (-1)
              #mat_sort = x[x[:,1].argsort()[::-1]]
              #mat_sort = x[x[:,2].argsort()[::-1]]
  
  
              #------------------------------------------------------------------------------
              #    Number of nearest Neighbors
              #------------------------------------------------------------------------------
              n_neighbors = 3
              #------------------------------------------------------------------------------
  
              dist_inv_total_knn = 0
              por_total_knn      = 0
              rhomatrix_total_knn      = 0
  
  
  
              #kNN Estimates for first 3 rows
              #dist_inv_total = mat_sort[0][0] + mat_sort[1][0] + mat_sort[2][0]
              for i in range(0,n_neighbors,1):
                  dist_inv_total_knn = dist_inv_total_knn + mat_sort[i][0]
                  por_total_knn  = por_total_knn + mat_sort[i][1]
                  rhomatrix_total_knn  = rhomatrix_total_knn + mat_sort[i][2]
  
  
              #back to k values and calculate estimations now
              por_est_knn  = por_total_knn  / dist_inv_total_knn
              rhomatrix_est_knn  = rhomatrix_total_knn  / dist_inv_total_knn
  
  
      #------------------------------------------------------------------------------ 
      #            Write Data to arrays
      #------------------------------------------------------------------------------
              #deptharray.append(Dep[k]);          # Taken from logs
              porarray.append(por_est_knn);       # Calculated Chartbook Porosity 
              rhoarray.append(rhomatrix_est_knn); # Calculated Chartbook Rhomatrix
  
      #logs['PHIT']=(porarray)
      #print(len(porarray))
      # Input parameters for Top and Bottom depths 
      #logs=data[(data.DEPT >= top_depth) & (data.DEPT <= bottom_depth)]
      #phit=porarray[(data.DEPT >= top_depth) & (data.DEPT <= bottom_depth)]
      #print(porarray)
      #print(data)
      return porarray,rhoarray

logs['PHIT'] , logs['RHOMAA'] = PHIT_Knn(logs['NPHI'].to_numpy(),logs['RHOB'].to_numpy())

print(logs['PHIT'])
method='Neut-Den'


if method == 'GR':
      logs['vsh'] = (logs['vshgr']).clip(0,1)
elif method == 'SP':
      logs['vsh'] = (logs['vshsp']).clip(0,1)
elif method == 'DT':
      logs['vsh'] = (logs['vshdt']).clip(0,1)
elif method == 'Neut-Den':
      logs['vsh'] = (logs['vshnd']).clip(0,1)
elif method == 'Neut-Mphi':
      logs['vsh'] = (logs['vshnmphi']).clip(0,1)
elif method == 'Median Filtered':
      logs['vsh'] = (logs['vshmedian']).clip(0,1)
elif method == 'Hodges-Lehmann':
      logs['vsh'] = (logs['vsh_hl']).clip(0,1)
  
def read_constant_from_file(file_path):
      try:
          with open(file_path, 'r') as file:
              constant = float(file.read().strip())
              return constant
      except FileNotFoundError:
          print(f"File '{file_path}' not found.")
      except ValueError:
          print(f"Invalid data in '{file_path}'. The file should contain only a single constant.")

 
# Replace 'new_cbw_int.txt' with the actual path to your file
file_path = './data/parameters/new_cbw_int.txt'
CBW_Int = read_constant_from_file(file_path)
  
if CBW_Int is not None:
      print("The CBW_Int from the file is:", CBW_Int)
  
  
print(CBW_Int)
  
logs['CBW'] = (logs['vsh'] * CBW_Int).clip(0,1)
  
logs['PHIE']=(logs['PHIT'] - CBW_Int*logs['vsh']).clip(0,1)# Calculations for Swb used in Dual Water and WaxSmits
logs['Swb'] =( 1 - logs['PHIE']/logs['PHIT']).clip(0,1)
  
# Qv from Swb using Hill, Shirley and Klein
logs['Qv'] = (logs['Swb']/(0.6425/((den_fl*SAL)**0.5) +0.22)).clip(0,5)


data1=logs
def Plot_sw2(m, n, rw, mslope,depth_range):

      logs = data1[(data1.DEPT >= depth_range[1]*-1) & (data1.DEPT <= depth_range[0]*-1)]


      ild = logs['ILD'].to_numpy()
      phit = logs['PHIT'].to_numpy()
      qv = logs['Qv'].to_numpy()
  
      bvw = []
      bvo = []
      swt = []
      mstarapp = []
  
      # Log Data
      for k in range(len(logs)):
          ILD = ild[k]
          PHIT = phit[k]
          QV = qv[k]
  
          # Saturation Calculations
          BVW = PHIT * ((1 / PHIT**m) * (rw / ILD))**(1/n)
          BVW = min(BVW, PHIT)
  
          Swt = BVW / PHIT
          BVO = PHIT * (1 - Swt)
          MSTARAPP = np.log10(rw / (ILD * (1 + rw * B * QV))) / np.log10(PHIT)
          
          bvo.append(BVO)
          swt.append(Swt)
          bvw.append(BVW)
          mstarapp.append(MSTARAPP)
  
      # Create subplots: 1 row, 3 columns
      fig = make_subplots(rows=1, cols=3, subplot_titles=('Bulk Volume Plot', 'Pickett Plot', 'Vsh vs. Mstar_Apparent'))
  
      y = logs['DEPT']
  
      # Bulk Volume Plot

      

      fig.add_trace(go.Scatter(x=logs['PHIT'], y=y, mode='lines', line=dict(color='red'), name='PHIT'), row=1, col=1)
      fig.add_trace(go.Scatter(x=logs['PHIE'], y=y, mode='lines', line=dict(color='orange'), name='PHIE'), row=1, col=1)


      #fig.add_trace(go.Scatter(x=bvo, y=y, fill='tonextx', mode='none', fillcolor='green', name='BVO'), row=1, col=1)
      fig.add_trace(go.Scatter(x=bvw, y=y, mode='lines', line=dict(color='black'), name='BVW', fill='tonextx', fillcolor='green'), row=1, col=1)
      
      fig.add_trace(go.Scatter(x=[0] * len(y), y=y,  fill='tonextx', mode='none', fillcolor='cyan', name='BVW'), row=1, col=1)
      
      fig.add_trace(go.Scatter(x=logs['PHIE'], y=y, mode='lines', line=dict(color='orange'), name='PHIE'), row=1, col=1)
       
      fig.update_xaxes(title_text="BVO/BVW", row=1, col=1, range=[0.5, 0])
      fig.update_yaxes(title_text="Depth", row=1, col=1, autorange="reversed")
  
      # Pickett Plot
      fig.add_trace(go.Scatter(x=logs['ILD'], y=logs['PHIT'], mode='markers', marker=dict(color='red'), name='Pickett'), row=1, col=2)
      fig.update_xaxes(title_text="ILD [ohmm]", type="log", range=[np.log10(0.01),np.log10(1000)], row=1, col=2)
      fig.update_yaxes(title_text="PHIT [v/v]", type="log", range=[np.log10(0.01),np.log10(1)], row=1, col=2)
  
      # Calculate saturation lines
      sw_plot = (1.0, 0.8, 0.6, 0.4, 0.2)
      phit_plot = (0.01, 1)
      rt_plot = np.zeros((len(sw_plot), len(phit_plot)))
  
      for i in range(len(sw_plot)):
          for j in range(len(phit_plot)):
              rt_result = (rw / (sw_plot[i]**n) * (1 / (phit_plot[j]**m)))
              rt_plot[i, j] = rt_result
          fig.add_trace(go.Scatter(x=rt_plot[i], y=phit_plot, mode='lines', line=dict(width=1), name=f'SW {int(sw_plot[i]*100)}%'), row=1, col=2)
  
      # Vsh vs. Mstar Apparent Plot
      fig.add_trace(go.Scatter(x=logs['vsh'], y=mstarapp, mode='markers', marker=dict(color='red'), name='Mstar Apparent'), row=1, col=3)
      fig.add_trace(go.Scatter(x=np.arange(10), y=np.arange(10) * mslope + m, mode='lines', line=dict(color='black'), name='Trend Line'), row=1, col=3)
      fig.update_xaxes(title_text="Vsh [v/v]", range=[0.0, 1], row=1, col=3)
      fig.update_yaxes(title_text="Mstar Apparent", range=[0, 7], row=1, col=3)
  
      fig.update_layout(title="Saturations from Logs", height=800, width=1000, showlegend=False)
  
      # Save the plot as a PNG file if needed
      #fig.write_image('phit_buck_plot_plotly.png', scale=2)
  
      # Display the plot
      #fig.show()
      #pio.write_html(fig, file='phit_buck_plot.html', auto_open=False, include_plotlyjs='cdn')
  
      return fig
  




def shale_plot3(sp_clean, sp_shale, gr_clean, gr_shale, neut_shale, dt_matrix, dt_shale,depth_range):
      print(depth_range)
      #logs = data[(data.DEPT >= depth_range[1]*-1) & (data.DEPT <= depth_range[0]*-1)]
      logs = data1[(data1.DEPT >= depth_range[1]*-1) & (data1.DEPT <= depth_range[0]*-1)]

      # Create a subplot with 7 rows and shared Y axis between depth tracks
      fig = make_subplots(
          rows=4, cols=4,
          shared_yaxes=True,
          column_widths=[0.25, 0.25, 0.25, 0.25],
          row_heights=[0.25, 0.25, 0.25, 0.25],
          horizontal_spacing=0.05,
          vertical_spacing=0.05,
          specs=[
              [{'rowspan': 4}, {}, {'rowspan': 4}, {'rowspan': 4}],
              [None, {},  None, None],
              [None, {},  None, None],
              [None, {},  None, None]
          ],
          subplot_titles=("SP & GR vs Depth", "SP Histogram", "GR Histogram", "NPHI vs MPHI", "NPHI vs RHOB", "Calculated Shale Volumes", "MRIL Log")
      )
      
      ###########
      # Plot GR and SP curves on the first subplot (leftmost)
      fig.add_trace(go.Scatter(x=logs['GR'], y=logs['DEPT'], mode='lines', name='GR', line=dict(color='green', width=1)), row=1, col=1)
      fig.add_trace(go.Scatter(x=logs['SP'], y=logs['DEPT'], mode='lines', name='SP', line=dict(color='black', width=1)), row=1, col=1)
      
      # Add vertical lines for clean/shale values
      fig.add_vline(x=gr_clean, line=dict(color='blue', width=1, dash='dash'), row=1, col=1)
      fig.add_vline(x=gr_shale, line=dict(color='brown', width=1, dash='dashdot'), row=1, col=1)
      fig.add_vline(x=sp_clean, line=dict(color='red', width=1), row=1, col=1)
      fig.add_vline(x=sp_shale, line=dict(color='gray', width=1), row=1, col=1)
  
      # Histograms for GR and SP
      fig.add_trace(go.Histogram(x=logs['GR'].dropna(), nbinsx=20, name='GR', marker_color='green'), row=2, col=2)
      fig.add_vline(x=gr_clean, line=dict(color='blue', width=1), row=2, col=2)
      fig.add_vline(x=gr_shale, line=dict(color='brown', width=1, dash='dashdot'), row=2, col=2)
  
      fig.add_trace(go.Histogram(x=logs['SP'].dropna(), nbinsx=20, name='SP', marker_color='black'), row=1, col=2)
      fig.add_vline(x=sp_clean, line=dict(color='red', width=1), row=1, col=2)
      fig.add_vline(x=sp_shale, line=dict(color='gray', width=1), row=1, col=2)
  
      # Crossplot NPHI vs MPHI
      #fig.add_trace(go.Scatter(x=logs['NPHI'], y=logs['MPHI'], mode='markers', name='NPHI vs MPHI', marker=dict(color='red', size=4)), row=3, col=2)
  
      # Crossplot NPHI vs RHOB
      fig.add_trace(go.Scatter(x=logs['NPHI'], y=logs['RHOB'], mode='markers', name='NPHI vs RHOB', marker=dict(color='red', size=4)), row=4, col=2)
      
      # Plot Shale Triangle N-Mphi
      fig.add_trace(go.Scatter(
      x=[neut_matrix, neut_fl, neut_shale, neut_matrix],
      y=[den_matrix,den_fl,den_shale,den_matrix],
      mode='lines+markers',
      marker=dict(color='blue'),
      name='Shale Triangle N-Mphi'), row=4, col=2)
      
      fig.update_yaxes(autorange='reversed', row=4, col=2)
      fig.update_yaxes(autorange='reversed', row=1, col=1)
      fig.update_yaxes(autorange='reversed', row=1, col=3)
      fig.update_yaxes(autorange='reversed', row=1, col=4)
      fig.update_xaxes(type='log', row=1, col=4)
  
      # Shale volume calculations and plot them
      fig.add_trace(go.Scatter(x=vshgr(logs['GR'], gr_clean, gr_shale), y=logs['DEPT'], mode='lines', name='Vsh_GR', line=dict(color='green', width=0.5)), row=1, col=3)
      fig.add_trace(go.Scatter(x=vshnd(logs['NPHI'], logs['RHOB'], neut_matrix, den_matrix, neut_fl, den_fl, neut_shale, den_shale), y=logs['DEPT'], mode='lines', name='Vsh_ND', line=dict(color='red', width=0.5)), row=1, col=3)
      fig.add_trace(go.Scatter(x=vshsp(logs['SP'], sp_clean, sp_shale), y=logs['DEPT'], mode='lines', name='Vsh_SP', line=dict(color='black', width=0.5)), row=1, col=3)
      fig.add_trace(go.Scatter(x=vshdt(logs['DT'], dt_matrix, dt_shale), y=logs['DEPT'], mode='lines', name='Vsh_DT', line=dict(color='blue', width=0.5)), row=1, col=3)
      #fig.add_trace(go.Scatter(x=logs['vshnmphi'],  y=logs['DEPT'], mode='lines', name='Vsh_NPHI', line=dict(color='orange', width=0.5)), row=1, col=3)

    #fig.add_trace(go.Scatter(x=vshdtden(logs['DT'], logs['RHOB'], dt_matrix,den_matrix,  dt_fluid,den_fl,  dt_shale,den_shale), y=logs['DEPT'], mode='lines',name='Vsh_DT_Den',line=dict(color='olive', width=0.5)), row=1, col=3)
      fig.add_trace(go.Scatter(x=logs['vsh_hl'], y=logs['DEPT'], mode='lines', name='Vsh Hodg-Lehman', line=dict(color='cyan', width=1)), row=1, col=3)
      fig.add_trace(go.Scatter(x=logs['vshmedian'], y=logs['DEPT'], mode='lines', name='Vsh Median', line=dict(color='purple', width=0.5)), row=1, col=3)

      #fig.add_trace(go.Scatter(x=logs['DT'], y=logs['DEPT'], mode='lines', name='DT', line=dict(color='green', width=1)), row=1, col=4)
      #fig.add_vline(x=dt_shale, line=dict(color='blue', width=1, dash='dash'), row=1, col=4)
      #fig.add_vline(x=dt_matrix, line=dict(color='blue', width=1, dash='dash'), row=1, col=4)

      fig.add_trace(go.Scatter(x=logs['ILD'], y=logs['DEPT'], mode='lines', name='ILD', line=dict(color='green', width=1)), row=1, col=4)
      #fig.add_vline(x=rt_shale, line=dict(color='blue', width=1, dash='dash'), row=1, col=4)
      #fig.add_vline(x=rt_clean, line=dict(color='blue', width=1, dash='dash'), row=1, col=4)


      # Add title and layout configurations
      fig.update_layout(
          height=800, width=1000, 
          title_text="Shale Parameter Plot: Volume of Shale from Different Methods",
          xaxis_title="Parameter Values",
          yaxis_title="Depth (ft)",
          showlegend=False
      )
      #fig.show()
      #pio.write_html(fig, file='Shale Parameter.html', auto_open=False, include_plotlyjs='cdn')
     
      return fig






# Initialize Dash app
dash_app = Dash(__name__)

# Define initial values for sliders
# GR endpoint parameters for shale calculations
gr_clean, gr_shale      = 45  , 120                             # Shale Parmaetrs for GR
  
  # SP endpoint parameters for shale calculations
sp_clean, sp_shale      = -50  , 0                              # Shale Parameters for SP
  
  # Endpoint Parameters for Neutron-Density Shale Triangle
neut_shale, den_shale   = 0.32 , 2.65                           # Shale Parmaters for Neutron-Density
neut_matrix, den_matrix = -0.04 , 2.65                          # Matrix Parmameters for Neutron-Density
neut_fl, den_fl         =  1.0 , 1.1                            # Fluid Parmameters for Neutron-Density
  
  # Fluid Parameters for MRIL tool
mphi_shale              = 0.0                                   # MRIL Parameters
  
  # Fluid Parameters for Sonic tool
dt_matrix, dt_fluid, dt_shale, cp, alpha = 55.5,188.0,90.0,1,5/8 # Sonic Parameters
  
  # Rt Shale parameters
rt_shale = 6
rt_clean = 170
  
m_cem, n_sat, mslope, Rw = 1.9, 2.0, 1.5, 0.15  # Fluid Parameters

# Define the Dash layout with reduced font size and aligned sliders
dash_app.layout = html.Div([
    html.Div([
        html.Div([
            # Shale parameters sliders
            html.Label('SP Clean', style={'font-size': '12px'}),
            dcc.Slider(id='sp-clean-slider', min=-100, max=25, step=0.5, value=sp_clean,
                       marks={i: str(i) for i in range(-100, 25, 10)},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Label('SP Shale', style={'font-size': '12px'}),
            dcc.Slider(id='sp-shale-slider', min=-100, max=25, step=0.5, value=sp_shale,
                       marks={i: str(i) for i in range(-100, 25, 10)},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Label('DT Clean', style={'font-size': '12px'}),
            dcc.Slider(id='dt-clean-slider', min=40, max=160, step=1, value=dt_matrix,
                       marks={i: str(i) for i in range(40, 161, 10)},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Label('DT Shale', style={'font-size': '12px'}),
            dcc.Slider(id='dt-shale-slider', min=40, max=160, step=1, value=dt_shale,
                       marks={i: str(i) for i in range(40, 161, 10)},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Label('GR Clean', style={'font-size': '12px'}),
            dcc.Slider(id='gr-clean-slider', min=0, max=150, step=0.5, value=gr_clean,
                       marks={i: str(i) for i in range(0, 151, 50)},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Label('GR Shale', style={'font-size': '12px'}),
            dcc.Slider(id='gr-shale-slider', min=0, max=150, step=0.5, value=gr_shale,
                       marks={i: str(i) for i in range(0, 151, 50)},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Label('Neutron Shale', style={'font-size': '12px'}),
            dcc.Slider(id='neutron-shale-slider', min=0, max=0.6, step=0.01, value=neut_shale,
                       marks={round(i * 0.1, 2): str(round(i * 0.1, 2)) for i in range(0, 7)},
                       tooltip={"placement": "bottom", "always_visible": True}),
            
            # Additional fluid parameter sliders
            html.Label("Cementation Exponent 'm_cem':", style={'font-size': '12px'}),
            dcc.Slider(id='m-slider', min=1.00, max=3.00, step=0.1, value=m_cem,
                       marks={round(i, 2): str(round(i, 2)) for i in [1.0, 1.5, 2.0, 2.5, 3.0]},
                       tooltip={"placement": "bottom", "always_visible": True}),
                       
            html.Label("Saturation Exponent 'n_sat':", style={'font-size': '12px'}),
            dcc.Slider(id='n-slider', min=1.00, max=3.00, step=0.1, value=n_sat,
                       marks={round(i, 2): str(round(i, 2)) for i in [1.0, 1.5, 2.0, 2.5, 3.0]},
                       tooltip={"placement": "bottom", "always_visible": True}),
                       
            html.Label("Rw:", style={'font-size': '12px'}),
            dcc.Slider(id='rw-slider', min=0.01, max=0.2, step=0.01, value=Rw,
                       marks={round(i, 2): str(round(i, 2)) for i in [0.01, 0.05, 0.1,0.15,0.2]},
                       tooltip={"placement": "bottom", "always_visible": True}),
                       
            html.Label("m* Slope:", style={'font-size': '12px'}),
            dcc.Slider(id='mslope-slider', min=0.01, max=4.00, step=0.1, value=mslope,
                       marks={round(i, 2): str(round(i, 2)) for i in [0.01, 1.0, 2.0, 3.0, 4.0]},
                       tooltip={"placement": "bottom", "always_visible": True}),
                       
            html.Label("CBW_Int:", style={'font-size': '12px'}),
            dcc.Slider(id='CBW_Int-slider', min=0.0, max=0.5, step=0.01, value=CBW_Int,
                       marks={round(i, 2): str(round(i, 2)) for i in [0.00, 0.1, 0.2, 0.3, 0.4]},
                       tooltip={"placement": "bottom", "always_visible": True}),
            
        ], style={'width': '250px', 'margin-top': '5px'}),  # Left side with sliders

        html.Label('Vsh Method:'),
        dcc.Dropdown(id='vsh-dropdown',
          options=[
            {'label': "GR", 'value': "GR"},
            {'label': "SP", 'value': "SP"},
            {'label': "DT", 'value': "DT"},
            {'label': "Neut-Den", 'value': "Neut-Den"},
            {'label': "Dt_Den", 'value': "Dt_Den"},
            {'label': "Neut-Mphi", 'value': "Neut-Mphi"},
            {'label': "Median Filtered", 'value': "Median Filtered"},
            {'label': "Hodges-Lehmann", 'value': "Hodges-Lehmann"},
            {'label': "Optimization", 'value': "Optimization"}
          ],
          value="Hodges-Lehmann",
          style={'width': '60%'}
        ),
        html.Div(id='output'),
        
        html.Button('Recalculate', id='recalculate-button', n_clicks=0, style={'margin-top': '10px'})
    ], style={'width': '60%'}),

    
   

    # Vertical Depth Range Slider with rotated label, placed next to horizontal sliders
    html.Div([
        dcc.RangeSlider(
            id='depth-range-slider',
            min=-1 * max_depth1,
            max=--1 * min_depth1
,
            step=1,
            value=[-1 * min_depth1, -1 * max_depth1],
            marks={i: str(i) for i in range(-1 * min_depth1, -1 * max_depth1, -100)},
            tooltip={"placement": "bottom", "always_visible": True},
            vertical=True,
            verticalHeight=900 
        ),
        html.Div(id='depth-range-display', style={'margin-left': '5px'})
    ], style={
        'height': '1000px',  
        'width': '100px',  
        'margin-top': '5px',  
        #'display': 'flex',
        'flex-direction': 'col',
        'align-items': 'left',
    }),
    
    html.Div([
        dcc.Graph(id='shale-plot')  # This is where shale_plot3 will be displayed
    ], style={'width': '100%', 'padding-left': '10px',}),

    html.Div([
        dcc.Graph(id='shale-plot1')  # This is where shale_plot3 will be displayed
    ], style={'display': 'flex','width': '100%', 'padding-left': '10px',}),


    # Shutdown button
    html.Button('Shutdown', id='shutdown-btn', n_clicks=0),
], style={'display': 'flex','width': '100%'})


@dash_app.callback(
    Output('output', 'children'),
    Input('vsh-dropdown', 'value')
)
def update_output(selected_vsh):
    return f'Selected Vsh Method: {selected_vsh}'
    print(selected_vsh)

# Callback to update depth range label and filter logs data
#@dash_app.callback(
    #Output('logs', 'children'),
    #Input('depth-range-slider', 'value')
#)
#def update_depth(depth_range):  
    #logs = data[(data.DEPT >= depth_range[0]*-1) & (data.DEPT <= depth_range[1]*-1)]

    
    #return logs

# Callback to update the plot based on the slider values
@dash_app.callback(
    Output('shale-plot', 'figure'),
    [Input('sp-clean-slider', 'value'),
     Input('sp-shale-slider', 'value'),
     Input('dt-clean-slider', 'value'),
     Input('dt-shale-slider', 'value'),
     Input('gr-clean-slider', 'value'),
     Input('gr-shale-slider', 'value'),
     Input('neutron-shale-slider', 'value'),
     
     Input('depth-range-slider', 'value')]
)
def update_plot(sp_clean, sp_shale, dt_clean, dt_shale, gr_clean, gr_shale, neut_shale,depth_range):
    

    fig = shale_plot3(sp_clean, sp_shale, gr_clean, gr_shale, neut_shale, dt_clean, dt_shale,depth_range)
    return fig






# Callback to shut down the Dash app
@dash_app.callback(
    Output('shutdown-btn', 'children'),
    [Input('shutdown-btn', 'n_clicks')]
)
def shutdown_server(n_clicks):
    if n_clicks > 0:
        os._exit(0)
    return "Shutdown Server"


# Callbacks to update labels
@dash_app.callback(
     Output('shale-plot1', 'figure'),

    
    [Input('m-slider', 'value'),
     Input('n-slider', 'value'),
     Input('rw-slider', 'value'),
     Input('mslope-slider', 'value'),
     Input('CBW_Int-slider', 'value'),
     Input('vsh-dropdown', 'value'),
     Input('depth-range-slider', 'value')]
)
def update_slider_labels(m, n, rw, mslope,CBW_Int,selected_vsh,depth_range):
    fig = Plot_sw2(m, n, rw, mslope,depth_range)
    return fig



# PyQt5 main application class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Dash in PyQt5')
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()
        self.start_button = QPushButton("Start Dash Server")
        layout.addWidget(self.start_button)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.start_button.clicked.connect(self.run_dash_server)

    def run_dash_server(self):
        threading.Thread(target=dash_app.run_server, kwargs={"port": 8050}).start()


# Initialize the Qt application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
