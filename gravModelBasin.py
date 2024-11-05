# -*- coding: utf-8 -*-
'''
Title: Basin Depth Modeling
Desc: Modeling basin depth from gravity data using a single density contrast.
    Uses the methodologies of Bott (1960) and Plouff (1976).
    Coordinates should be in UTM (m) and gravity in mGal.
    Using multiprocessing will speed up the model but has issues on Windows.
Author: Bradford Mack
Date: 12 Dec 23
Last modified: 5 Nov 24
'''

from datetime import datetime as dt
import gravModelLib as gm
from itertools import repeat
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pygmt as pg
from scipy.interpolate import griddata
from tqdm import tqdm
import xarray as xr
import yaml


#%% Load inputs

paramfile = 'BasinModelParams.txt'

with open(paramfile, 'r') as yml:
    params = yaml.safe_load(yml)


mainSaveDir = params['Main Save Directory'] # main project folder to save all data and figures. Code will create subfolders in this directory. Must have / at the end
dfGravData = pd.read_csv(mainSaveDir+params['Data File'],header=0) # Import grav data. Must have 4 columns: Easting (m), Northing (m), Elevation (m), and Bouguer Anomaly (mGal)
if 'Profile Coordinates File' in params:
    dfProfileCoord = pd.read_csv(mainSaveDir+params['Profile Coordinates File'],header=0) # Import profile coordinates. Must have 5 columns: Profile Number, Start X (m), Start Y (m), End X (m), End Y (m)

rho = params['Density Contrast'] # density contrast in kg/m^3
gridSpace = params['Grid Spacing'] # spacing of interpolation grid in meters
maxDepth = params['Maximum Depth'] # maximum basin depth in m
maskRad = params['Masking Radius'] # search radius around grav points to determine what is plotted on maps (masking). Use 0 for no masking

useBlockMed = params['Use Block Median'] # determines whether to use block median before interpolation of grav data. True = use block median. Helps smooth interpolation
useMP = params['Use MultiProcessing'] # determines whether to use multiprocessing for the 3D model. MP has issues on Windows and may not work correctly. If the progress bar freezes, it is not working and you must use the other method (useMP=False)

z0 = params['Station Depth'] # depth of gravity staion. 0 is at the surface
prismTop = params['Prism Top'] # depth of top of prism in m. Should not be 0.

intMethod = params['Interpolation Method'] # method for gravity interpolation. Can use 'nearest', 'linear', or 'cubic'

southBound = params['South Boundary'] # northing of southern boundary cutoff for modeling
northBound = params['North Boundary'] # northing of northern boundary cutoff for modeling
westBound = params['West Boundary'] # easting of western boundary cutoff for modeling
eastBound = params['East Boundary'] # easting of eastern boundary cutoff for modeling

if 'Profile Numbers' in params:
    profileNums = params['Profile Numbers'] # list of profile numbers to plot on observed gravity map. Numbers correspond to those in profile coordinate csv

del(params,yml)


#%% 2: Error handling

assert type(gridSpace) == int or type(gridSpace) == float, 'Grid spacing must be integer or float'
assert type(rho) == int or type(rho) == float, 'Density contrast must be integer or float'
assert type(z0) == int or type(z0) == float, 'Depth of gravity station must be integer or float'
assert type(maxDepth) == int or type(maxDepth) == float, 'Max depth must be integer or float'
assert type(prismTop) == int or type(prismTop) == float, 'Prism top depth must be integer or float'
assert type(maskRad) == int or type(maskRad) == float, 'Mask radius must be integer or float'
assert type(southBound) == int or type(southBound) == float,'Boundaries must be integer or float'
assert type(northBound) == int or type(northBound) == float,'Boundaries must be integer or float'
assert type(westBound) == int or type(westBound) == float,'Boundaries must be integer or float'
assert type(eastBound) == int or type(eastBound) == float,'Boundaries must be integer or float'
assert type(useBlockMed) == bool, 'useBlockMed must be a boolean'
assert type(useMP) == bool, 'useMP must be a boolean'
if 'profileNums' in globals():
    assert type(profileNums) == list, 'Profile numbers must be given as a list'

assert gridSpace > 0, 'Grid spacing must be > 0'
assert rho < 0, 'Density contrast must be < 0'
assert z0 <= 0, 'Depth of gravity station must be <= 0 (depth is positive down)'
assert maxDepth > 0, 'Max depth must be > 0'
assert prismTop > 0, 'Prism top depth must be > 0'
assert maskRad >= 0, 'Mask radius must be >= 0'
assert southBound >= 0, 'Boundaries cannot be negative'
assert northBound >= 0, 'Boundaries cannot be negative'
assert westBound >= 0, 'Boundaries cannot be negative'
assert eastBound >= 0, 'Boundaries cannot be negative'
assert intMethod == 'cubic' or intMethod == 'linear' or intMethod == 'nearest','Interpolation method must be nearest, linear, or cubic'
assert len(dfGravData.columns) == 4, 'Grav data must have 4 columns: Easting (m), Northing (m), Elevation (m), and Bouguer Anomaly (mGal)'
if 'dfProfileCoord' in globals():
    assert len(dfProfileCoord.columns) == 5, 'Profile coordinates file must have 5 columns: Profile Number, Start X (m), Start Y (m), End X (m), End Y (m)'


#%% 3: Calculate map boundaries based on given boundaries and grid spacing

southBoundRnd = gm.roundDown(southBound, gridSpace) # northing of southern boundary cutoff for data
eastBoundRnd = gm.roundUp(eastBound, gridSpace) # easting of eastern boundary cutoff for data
northBoundRnd = gm.roundUp(northBound, gridSpace) # northing of northern boundary cutoff for data
westBoundRnd = gm.roundDown(westBound, gridSpace) # easting of western boundary cutoff for data


#%% 4: Interpolate/grid gravity data

saveDir = mainSaveDir+'GriddedGravData/'+str(gridSpace)+'m/' # directory to save gridded/interpolated data in

if not os.path.isdir(saveDir): # create directory if it doesn't exist
    os.makedirs(saveDir)


if useBlockMed == True: # use block median before gridding/interpolation

    dfBlockMed = pg.blockmedian(x=dfGravData.iloc[:,0],y=dfGravData.iloc[:,1],
                                z=dfGravData.iloc[:,3], region=[min(dfGravData.iloc[:,0]),
                                max(dfGravData.iloc[:,0]),min(dfGravData.iloc[:,1]),
                                max(dfGravData.iloc[:,1])], spacing=str(gridSpace)+'+e') # create dataframe of grav data with block median calculated
    
    eastMin = gm.roundDown(min(dfBlockMed.iloc[:,0])-gridSpace,gridSpace) # find minimum Easting value in m
    eastMax = gm.roundUp(max(dfBlockMed.iloc[:,0])+gridSpace,gridSpace) # find maximum Easting value in m
    northMin = gm.roundDown(min(dfBlockMed.iloc[:,1])-gridSpace,gridSpace) # find minimum Northing value in m
    northMax = gm.roundUp(max(dfBlockMed.iloc[:,1])+gridSpace,gridSpace) # find maximum Northing value in m
    
    gridEast, gridNorth = np.mgrid[eastMin:eastMax+gridSpace:gridSpace,
                                   northMin:northMax+gridSpace:gridSpace] # create 2D arrays of Easting and Northing values for each observation point
    
    eastArray = dfBlockMed.iloc[:,0].values # create numpy array of Easting values
    northArray = dfBlockMed.iloc[:,1].values # create numpy array of Northing values
    coordinates = np.stack([eastArray,northArray]).transpose() # merge Easting and Northing arrays into single array of coordinate pairs
    
    gridGrav = griddata(coordinates, dfBlockMed.iloc[:,2], (gridEast,gridNorth), method=intMethod) # interpolate gridded gravity data
    
    eastList = [] # create empty list to store all Easting values to be put into gridded dataframe
    northList = [] # create empty list to store all Northing values to be put into gridded dataframe
    gravList = [] # create empty list to store all gravity anomaly values to be put into gridded dataframe
     
    for i in range(len(gridEast)): # loop through rows of gridded coordinates
        for j in range(len(gridEast[0])): # loop through columns of gridded coordinates
            eastList.append(gridEast[i,j]) # store Easting value to list
            northList.append(gridNorth[i,j]) # store Northing value to list
            gravList.append(gridGrav[i,j]) # store gravity anomaly value to list
             
    zipped = list(zip(eastList,northList,gravList)) # zip 3 lists of coordinates and gravity values together
    
    dfBlockMedGrid = pd.DataFrame(zipped, columns=['Easting (m)','Northing (m)','Gravity Anomaly (mGal)']) # create dataframe of gridded gravity data
    
    dfBlockMedGrid.to_csv(saveDir+'griddedGravAll_'+str(gridSpace)+'mNaN_BlockMed.csv',index=False) # save dataframe to CSV with NaN
    
    dfBlockMedGrid.dropna(inplace=True,ignore_index=True) # remove rows with NaN values in gravity column
    dfBlockMedGrid.to_csv(saveDir+'griddedGravAll_'+str(gridSpace)+'m_BlockMed.csv',index=False) # save dataframe to CSV without NaN

    dfBlockMedGrid = dfBlockMedGrid[dfBlockMedGrid.iloc[:,1] >= southBoundRnd] # drop rows outside of south boundary
    dfBlockMedGrid = dfBlockMedGrid[dfBlockMedGrid.iloc[:,0] <= eastBoundRnd] # drop rows outside of east boundary
    dfBlockMedGrid = dfBlockMedGrid[dfBlockMedGrid.iloc[:,1] <= northBoundRnd] # drop rows outside of north boundary
    dfBlockMedGrid = dfBlockMedGrid[dfBlockMedGrid.iloc[:,0] >= westBoundRnd] # drop rows outside of west boundary
    
    dfBlockMedGrid.reset_index(drop=True,inplace=True) # reset dataframe indices after dropping rows
    dfBlockMedGrid.to_csv(saveDir+'griddedGravAll_'+str(gridSpace)+'mTrim_BlockMed.csv',index=False) # save trimmed dataframe to CSV 
    
    dfGravDataGrid = dfBlockMedGrid.copy()
    
    del(coordinates,dfBlockMed,dfBlockMedGrid,eastArray,eastBound,eastList,
        eastMax,eastMin,gravList,gridEast,gridGrav,gridNorth,i,j,northArray,northBound,
        northList,northMax,northMin,southBound,westBound,zipped)
    

else: # don't use block median
    
    eastMin = gm.roundDown(min(dfGravData.iloc[:,0])-gridSpace,gridSpace) # find minimum Easting value in m
    eastMax = gm.roundUp(max(dfGravData.iloc[:,0])+gridSpace,gridSpace) # find maximum Easting value in m
    northMin = gm.roundDown(min(dfGravData.iloc[:,1])-gridSpace,gridSpace) # find minimum Northing value in m
    northMax = gm.roundUp(max(dfGravData.iloc[:,1])+gridSpace,gridSpace) # find maximum Northing value in m
    
    gridEast, gridNorth = np.mgrid[eastMin:eastMax+gridSpace:gridSpace,
                                   northMin:northMax+gridSpace:gridSpace] # create 2D arrays of Easting and Northing values for each observation point
    
    eastArray = dfGravData.iloc[:,0].values # create numpy array of Easting values
    northArray = dfGravData.iloc[:,1].values # create numpy array of Northing values
    coordinates = np.stack([eastArray,northArray]).transpose() # merge Easting and Northing arrays into single array of coordinate pairs
       
    gridGrav = griddata(coordinates, dfGravData.iloc[:,3], (gridEast,gridNorth), method=intMethod) # interpolate gridded gravity data

    eastList = [] # create empty list to store all Easting values to be put into gridded dataframe
    northList = [] # create empty list to store all Northing values to be put into gridded dataframe
    gravList = [] # create empty list to store all gravity anomaly values to be put into gridded dataframe
    
    for i in range(len(gridEast)): # loop through rows of gridded coordinates
        for j in range(len(gridEast[0])): # loop through columns of gridded coordinates
            eastList.append(gridEast[i,j]) # store Easting value to list
            northList.append(gridNorth[i,j]) # store Northing value to list
            gravList.append(gridGrav[i,j]) # store gravity anomaly value to list
            
    zipped = list(zip(eastList,northList,gravList)) # zip 3 lists of coordinates and gravity values together
    
    dfGravDataGrid = pd.DataFrame(zipped, columns=['Easting (m)','Northing (m)','Gravity Anomaly (mGal)']) # create dataframe of gridded gravity data
    
    dfGravDataGrid.to_csv(saveDir+'griddedGravAll_'+str(gridSpace)+'mNaN.csv',index=False) # save dataframe to CSV with NaN
    
    dfGravDataGrid.dropna(inplace=True,ignore_index=True) # remove rows with NaN values in gravity column
    dfGravDataGrid.to_csv(saveDir+'griddedGravAll_'+str(gridSpace)+'m.csv',index=False) # save dataframe to CSV without NaN
    
    dfGravDataGrid = dfGravDataGrid[dfGravDataGrid.iloc[:,1] >= southBoundRnd] # drop rows outside of south boundary
    dfGravDataGrid = dfGravDataGrid[dfGravDataGrid.iloc[:,0] <= eastBoundRnd] # drop rows outside of east boundary
    dfGravDataGrid = dfGravDataGrid[dfGravDataGrid.iloc[:,1] <= northBoundRnd] # drop rows outside of north boundary
    dfGravDataGrid = dfGravDataGrid[dfGravDataGrid.iloc[:,0] >= westBoundRnd] # drop rows outside of west boundary
    dfGravDataGrid.reset_index(drop=True,inplace=True) # reset dataframe indices after dropping rows
    dfGravDataGrid.to_csv(saveDir+'griddedGravAll_'+str(gridSpace)+'mTrim.csv',index=False) # save trimmed dataframe to CSV 
    
    
    del(coordinates,eastArray,eastBound,eastList,eastMax,eastMin,gravList,
        gridEast,gridGrav,gridNorth,i,j,northArray,northBound,northList,northMax,
        northMin,saveDir,southBound,westBound,zipped)


#%% 5: Calculate basin depth, calculated gravity from depth, and difference in gravity

if useMP == True: # use multiprocessing
    numCPU = mp.cpu_count() # returns total number of CPUs
    
    dfGravDataGrid = gm.bouguerDF(dfGravDataGrid,rho) # calculate initial basin thickness. Adds thickness column to df which also become dfPrism column 7
    
    saveDir = mainSaveDir+'3dModelData/'+str(gridSpace)+'m/'
    if not os.path.isdir(saveDir): # create directory if it doesn't exist
        os.makedirs(saveDir)
    if useBlockMed == True:
        saveFileName = '3dData_'+str(gridSpace)+'m'+str(rho)+'_initial_BlockMed.csv' # file name for saving df to csv
    else:
        saveFileName = '3dData_'+str(gridSpace)+'m'+str(rho)+'_initial.csv' # file name for saving df to csv
    
    dfGravDataGrid.to_csv(saveDir+saveFileName, index=False) # save CSV of initial values
    
    
    dfPrism = dfGravDataGrid.copy() # copy gravity dataframe to use for calculating prisms
    dfPrism.drop(dfPrism.columns[2],axis=1,inplace=True) # remove gravity anomaly column
    dfPrism.insert(loc=2,column='W Boundary (m)',value=0) # add column for western prism boundary
    dfPrism.insert(loc=3,column='S Boundary (m)',value=0) # add column for southern prism boundary
    dfPrism.insert(loc=4,column='Prism Top (m)',value=prismTop) # add column for prism top
    dfPrism.insert(loc=5,column='E Boundary (m)',value=0) # add column for eastern prism boundary
    dfPrism.insert(loc=6,column='N Boundary (m)',value=0) # add column for northern prism boundary. 2-6 were inserted before thickness column, so it gets pushed to 7
    dfPrism.insert(loc=8,column='Density Contrast (kg/m^3',value=rho) # add column for density contrast
    
        
    for i in range(len(dfPrism)): # Calculate boundaries of each prism
        dfPrism.iloc[i,2] = dfPrism.iloc[i,0]-gridSpace/2 # west edge of prism in m
        dfPrism.iloc[i,3] = dfPrism.iloc[i,1]-gridSpace/2 # south edge of prism in m
        dfPrism.iloc[i,5] = dfPrism.iloc[i,0]+gridSpace/2 # east edge of prism in m
        dfPrism.iloc[i,6] = dfPrism.iloc[i,1]+gridSpace/2 # north edge of prism in m
          
    
    dfGravDataGrid.insert(loc=4,column='Calc Grav (mGal)',value=0.0) # add column to store calculated gravity from thickness values
    dfGravDataGrid.insert(loc=5,column='Diff Grav (mGal)',value=0.0) # add column to store difference between observed and calculated gravity values
    dfGravDataGrid.insert(loc=6,column='d Thick (m)',value=0.0) # add column to store change in thickness values

    
    # !!! Some pooling issue causing it to hang inside the tqdm loop in Windows. Works in Linux. If the progress bar freezes, it's not working. Set useMP = False to use other version
    for step in range(10): # run prism calculation multiple times and adjust thickness to minimize obs-calc error
        for i in tqdm(range(len(dfGravDataGrid))): # iterate through all observation points on the grid
            pool = mp.Pool(numCPU) # create pool object with max child processes = number of CPUs available
    
            calcGravList = pool.starmap(gm.gravPrism,zip(repeat(dfGravDataGrid.iloc[i,0]), # calculate gravity at each OP as a sum of all prisms
                        repeat(dfGravDataGrid.iloc[i,1]),repeat(z0),
                        dfPrism.iloc[:,2].to_list(),dfPrism.iloc[:,3].to_list(),
                        dfPrism.iloc[:,4].to_list(),dfPrism.iloc[:,5].to_list(),
                        dfPrism.iloc[:,6].to_list(),dfPrism.iloc[:,7].to_list(),
                        dfPrism.iloc[:,8].to_list()))
            
            dfGravDataGrid.iloc[i,4] = sum(calcGravList) # store calculated gravity value in dataframe
            dfGravDataGrid.iloc[i,5] = dfGravDataGrid.iloc[i,2]-dfGravDataGrid.iloc[i,4] # find the difference between calculated and observed gravity    
            dThick = gm.bouguer(dfGravDataGrid.iloc[i,5],rho) # calculate difference in thickness due to difference in gravity
            dfGravDataGrid.iloc[i,6] = dThick
            dfGravDataGrid.iloc[i,3] += dThick #adjust thickness for final maps
            dfPrism.iloc[i,7] += dThick #adjust thickness for next iteration of prism calculations
            if dfGravDataGrid.iloc[i,3] < 0: # do not allow negative thickness
                dfGravDataGrid.iloc[i,3] = 0
            if dfGravDataGrid.iloc[i,3] > maxDepth: # do not exceed max basin depth
                dfGravDataGrid.iloc[i,3] = maxDepth
            if dfPrism.iloc[i,7] < 0: # do not allow negative thickness
                dfPrism.iloc[i,7] = 0
            if dfPrism.iloc[i,7] > maxDepth: # do not exceed max basin depth
                dfPrism.iloc[i,7] = maxDepth
                
        print('Step ',step,' complete. '+ str(dt.now().time()))
        
        if useBlockMed == True:
            saveFileName = '3dData_'+str(gridSpace)+'m'+str(rho)+'_step'+str(step)+'MP_BlockMed.csv' # file name for saving df to csv
        else:
            saveFileName = '3dData_'+str(gridSpace)+'m'+str(rho)+'_step'+str(step)+'MP.csv' # file name for saving df to csv
            
        dfGravDataGrid.to_csv(saveDir+saveFileName, index=False) # save dataframe to csv


    del(calcGravList,dfPrism,dThick,i,numCPU,pool,saveDir,saveFileName,step)    



else: # no multiprocessing
    dfGravDataGrid = gm.bouguerDF(dfGravDataGrid,rho) # calculate initial basin thickness. Adds thickness column to df which also become dfPrism column 7
    
    saveDir = mainSaveDir+'3dModelData/'+str(gridSpace)+'m/'
    if not os.path.isdir(saveDir): # create directory if it doesn't exist
        os.makedirs(saveDir)
    if useBlockMed == True:
        saveFileName = '3dData_'+str(gridSpace)+'m'+str(rho)+'_initial_BlockMed.csv' # file name for saving df to csv
    else:
        saveFileName = '3dData_'+str(gridSpace)+'m'+str(rho)+'_initial.csv' # file name for saving df to csv
    
    dfGravDataGrid.to_csv(saveFileName, index=False)
    
    
    dfPrism = dfGravDataGrid.copy() # copy gravity dataframe to use for calculating prisms
    dfPrism.drop(dfPrism.columns[2],axis=1,inplace=True) # remove gravity anomaly column
    dfPrism.insert(loc=2,column='W Boundary (m)',value=0) # add column for western prism boundary
    dfPrism.insert(loc=3,column='S Boundary (m)',value=0) # add column for southern prism boundary
    dfPrism.insert(loc=4,column='Prism Top (m)',value=prismTop) # add column for prism top
    dfPrism.insert(loc=5,column='E Boundary (m)',value=0) # add column for eastern prism boundary
    dfPrism.insert(loc=6,column='N Boundary (m)',value=0) # add column for northern prism boundary. 2-6 were inserted before thickness column, so it gets pushed to 7
    dfPrism.insert(loc=8,column='Density Contrast (kg/m^3',value=rho) # add column for density contrast
    
        
    for i in range(len(dfPrism)): # Calculate boundaries of each prism
        dfPrism.iloc[i,2] = dfPrism.iloc[i,0]-gridSpace/2 # west edge of prism in m
        dfPrism.iloc[i,3] = dfPrism.iloc[i,1]-gridSpace/2 # south edge of prism in m
        dfPrism.iloc[i,5] = dfPrism.iloc[i,0]+gridSpace/2 # east edge of prism in m
        dfPrism.iloc[i,6] = dfPrism.iloc[i,1]+gridSpace/2 # north edge of prism in m
          
    
    dfGravDataGrid.insert(loc=4,column='Calc Grav (mGal)',value=0.0) # add column to store calculated gravity from thickness values
    dfGravDataGrid.insert(loc=5,column='Diff Grav (mGal)',value=0.0) # add column to store difference between observed and calculated gravity values
    dfGravDataGrid.insert(loc=6,column='d Thick (m)',value=0.0) # add column to store calculated gravity from thickness values


    for step in range(10): # run prism calculation multiple times and adjust thickness to minimize obs-calc error
        for i in tqdm(range(len(dfGravDataGrid))): # iterate through all observation points on the grid
            dfGravDataGrid.iloc[i,4] = 0 # zero out calculated gravity before running next iteration
            for j in range(len(dfPrism)): # iterate over every prism and sum gravity at OP
                dfGravDataGrid.iloc[i,4] += gm.gravPrism(dfGravDataGrid.iloc[i,0],\
                        dfGravDataGrid.iloc[i,1],z0,dfPrism.iloc[j,2],dfPrism.iloc[j,3],\
                        dfPrism.iloc[j,4],dfPrism.iloc[j,5],dfPrism.iloc[j,6],\
                        dfPrism.iloc[j,7],dfPrism.iloc[j,8]) # add calculated gravity from prism to sum
          
            dfGravDataGrid.iloc[i,5] = dfGravDataGrid.iloc[i,2]-dfGravDataGrid.iloc[i,4] # find the difference between calculated and observed gravity
                 
            dThick = gm.bouguer(dfGravDataGrid.iloc[i,5],rho) # calculate difference in thickness due to difference in gravity
            dfGravDataGrid.iloc[i,6] = dThick # save change in thickness to dataframe
            dfGravDataGrid.iloc[i,3] += dThick #adjust thickness for final maps
            dfPrism.iloc[i,7] += dThick #adjust thickness for next iteration of prism calculations
        
            if dfGravDataGrid.iloc[i,3] < 0: # do not allow negative thickness
                dfGravDataGrid.iloc[i,3] = 0
            if dfGravDataGrid.iloc[i,3] > maxDepth: # do not exceed max basin depth
                dfGravDataGrid.iloc[i,3] = maxDepth
            if dfPrism.iloc[i,7] < 0: # do not allow negative thickness
                dfPrism.iloc[i,7] = 0
            if dfPrism.iloc[i,7] > maxDepth: # do not exceed max basin depth
                dfPrism.iloc[i,7] = maxDepth
                
        print('Step ',step,' complete. '+ str(dt.now().time()))
        
        if useBlockMed == True:
            saveFileName = '3dData_'+str(gridSpace)+'m'+str(rho)+'_step'+str(step)+'_BlockMed.csv' # file name for saving df to csv
        else:
            saveFileName = '3dData_'+str(gridSpace)+'m'+str(rho)+'_step'+str(step)+'.csv' # file name for saving df to csv
            
        dfGravDataGrid.to_csv(saveDir+saveFileName, index=False) # save dataframe to csv


    del(dfPrism,dThick,i,j,saveDir,step,useMP) 


#%%% 6: Grid masking

if 'dfGravDataGrid' not in globals(): # reads gridded grav data file if you want to skip to here and not rerun the model
    if useBlockMed == True:
        readFile = mainSaveDir+'3dModelData/'+str(gridSpace)+'m/3dData_'+str(gridSpace)+'m'+str(rho)+'_step9MP_BlockMed.csv'
    else:
        readFile = mainSaveDir+'3dModelData/'+str(gridSpace)+'m/3dData_'+str(gridSpace)+'m'+str(rho)+'_step9MP.csv'
    dfGravDataGrid = pd.read_csv(readFile,header=0)
    del(readFile)


dfMasked = dfGravDataGrid.copy() # copy gridded grav data to mask data for a given search radius
dfMasked.drop(dfMasked.columns[6],axis=1,inplace=True) # remove unnecessary columns

dfMasked.insert(loc=6,column='Within Radius',value=False) # add column to store boolean if observation point is within radius from any grav stations


if maskRad == 0:
    for i in range(len(dfMasked)):
        dfMasked.iloc[i,6] = True # mark OP as True
else:
    for i in tqdm(range(len(dfMasked))): # loop through all gridded observation points
        for j in range(len(dfGravData)): # loop through all gravity stations
            dist = np.sqrt((dfMasked.iloc[i,0]-dfGravData.iloc[j,0])**2+(dfMasked.iloc[i,1]-dfGravData.iloc[j,1])**2) # calculate distance between OP and station
            if dist <= maskRad: # check if distance is <= search radius
                dfMasked.iloc[i,6] = True # mark OP as True if it is with radius of ANY grav station
                break # exit current iteration of j loop and move on to next

for i in range(len(dfMasked)):
    if dfMasked.iloc[i,6] == False:
        dfMasked.iloc[i,2:6] = np.nan # change all values to NaN if not within search radius
dfMasked.dropna(inplace=True,ignore_index=True) # remove rows with NaN values
     
        
saveDir = mainSaveDir+'MaskedData/'+str(gridSpace)+'m/' # path to masked data directory

if not os.path.isdir(saveDir): # create directory if it doesn't exist
    os.makedirs(saveDir)
        
if useBlockMed == True:
    saveFileName = '3dData_'+str(gridSpace)+'m'+str(rho)+'Masked'+str(maskRad)+'_BlockMed.csv'
else:
    saveFileName = '3dData_'+str(gridSpace)+'m'+str(rho)+'Masked'+str(maskRad)+'.csv'

dfMasked.to_csv(saveDir+saveFileName, index=False) # save dataframe to csv
              

del(saveDir,saveFileName)
  

#%% 7: Create directory for saving figures

saveDir = mainSaveDir+'Figures/'+str(gridSpace)+'m/' # path to figure directory

if not os.path.isdir(saveDir): # create directory if it doesn't exist
    os.makedirs(saveDir)
    
 
#%% 8: Scale maps and load masked data if not loaded

xSize = 15 # x axis length in cm
yScale = (northBoundRnd-southBoundRnd)/(eastBoundRnd-westBoundRnd) # ratio of y axis size to x axis size
projStr = 'X'+str(xSize)+'c/'+str(yScale*xSize)+'c' # string for pygmt projection to make the map equal aspect

if 'dfMasked' not in globals(): # reads masked grav data file if you want to skip to here and not rerun the model or masking
    if useBlockMed == True:
        readFile = mainSaveDir+'MaskedData/'+str(gridSpace)+'m/3dData_'+str(gridSpace)+'m'+str(rho)+'Masked'+str(maskRad)+'_BlockMed.csv'
    else:
        readFile = mainSaveDir+'MaskedData/'+str(gridSpace)+'m/3dData_'+str(gridSpace)+'m'+str(rho)+'Masked'+str(maskRad)+'.csv'

    dfMasked = pd.read_csv(readFile,header=0)
    
    
    del(readFile)

regionList = [westBoundRnd,eastBoundRnd,southBoundRnd,northBoundRnd] # create list of map boundaries


#%%% 9: PyGMT thickness map plotting

thickGrid = pg.xyz2grd(x=dfMasked.iloc[:,0], y=dfMasked.iloc[:,1],
             z=dfMasked.iloc[:,3], spacing=gridSpace, region=regionList) # create grid file of basin thickness
    

fig = pg.Figure()

fig.grdimage(region=regionList,grid=thickGrid, cmap='dem2', dpi=300,\
             projection=projStr) # create contour fill. dpi controls smoothness
fig.grdcontour(grid=thickGrid,levels=250,annotation=1000,
             frame=['ag5000','+tThickness Map, Contrast='+ str(rho) + \
                    ', Masked='+str(maskRad)+' m, BlockMed='+str(useBlockMed),\
                        "xaf+lEasting (m)", "yaf+lNorthing (m)"]) # create contour lines. levels is contour interval. frame is border style and grid lines
fig.colorbar(frame=['a500f500','x+lDepth (m)'],  # Set annotations by 500 m
             position="JRM") # Place colorbar at position Right Middle

fig.show()


if useBlockMed == True:
    saveFileName = 'thickMap_'+str(gridSpace)+'m'+str(rho)+'Mask'+str(maskRad)+'m_BlockMed.png'
else:
    saveFileName = 'thickMap_'+str(gridSpace)+'m'+str(rho)+'Mask'+str(maskRad)+'m.png'

fig.savefig(saveDir+saveFileName) # save figure


del(fig,saveFileName)


#%% 10: PyGMT Observed gravity map plotting

obsGrid = pg.xyz2grd(x=dfMasked.iloc[:,0],y=dfMasked.iloc[:,1],
             z=dfMasked.iloc[:,2], spacing=gridSpace, region=regionList) # create grid file of observed gravity
    

fig = pg.Figure()

fig.grdimage(region=regionList,grid=obsGrid,cmap='haxby',dpi=300,\
             projection=projStr) # create contour fill. dpi controls smoothness
fig.grdcontour(grid=obsGrid,levels=5,annotation=10,
             frame=['ag5000','+tObserved Gravity Map, Contrast='+ str(rho) + \
                    ', Masked='+str(maskRad)+' m, BlockMed='+str(useBlockMed),\
                            "xaf+lEasting (m)", "yaf+lNorthing (m)"]) # create contour lines. levels is contour interval. frame is border style and grid lines

fig.plot(x=dfGravData.iloc[:,0], y=dfGravData.iloc[:,1], style='p0.05c') #plot grav station points

if 'dfProfileCoord' in globals(): # will not run this part if dfProfileCoord is commented out in inputs. Ignore undefined variable warnings if this is the case
    for i in range(len(dfProfileCoord)): # loop through profile coordinate dataframe
        for num in profileNums: # loop through profile numbers for plotting
            if dfProfileCoord.iloc[i,0] == num: # find profile data by number 
                fig.plot(x=[dfProfileCoord.iloc[i,1],dfProfileCoord.iloc[i,3]],\
                         y=[dfProfileCoord.iloc[i,2],dfProfileCoord.iloc[i,4]],pen="1p,red") # plot profile lines on map
                if dfProfileCoord.iloc[i,2] <= dfProfileCoord.iloc[i,4]: # if line runs toward north
                    yOff = (northBoundRnd-southBoundRnd)/40 # y offset of label from start point
                elif dfProfileCoord.iloc[i,2] > dfProfileCoord.iloc[i,4]: # if line runs toward south
                    yOff = (southBoundRnd-northBoundRnd)/40 # y offset of label from start point
                fig.text(x=dfProfileCoord.iloc[i,1],y=dfProfileCoord.iloc[i,2]-yOff,text='Profile '+str(num),font="10p,red") # label profile lines
    del(yOff)

fig.colorbar(frame=['a5f5','x+lGravity Anomaly (mGal)'],  # Set annotations by 5 mGal
             position="JRM") # Place colorbar at position Right Middle

fig.show()


if useBlockMed == True: 
    saveFileName = 'obsGravMap_'+str(gridSpace)+'m'+str(rho)+'Mask'+str(maskRad)+'m_BlockMed.png'
else:
    saveFileName = 'obsGravMap_'+str(gridSpace)+'m'+str(rho)+'Mask'+str(maskRad)+'m.png'

if 'dfProfileCoord' in globals(): # add file name tag for whether profiles are plotted
    saveFileName = saveFileName[:-4]+'_Pro.png'
else:
    saveFileName = saveFileName[:-4]+'_NoPro.png'

fig.savefig(saveDir+saveFileName) # save figure


del(fig,obsGrid,saveFileName)
    

#%% 11: PyGMT Calculated gravity map plotting

calcGrid = pg.xyz2grd(x=dfMasked.iloc[:,0],y=dfMasked.iloc[:,1],
             z=dfMasked.iloc[:,4], spacing=gridSpace, region=regionList) # create grid file of calculated gravity
    

fig = pg.Figure()

fig.grdimage(region=regionList,grid=calcGrid,cmap='haxby',dpi=300,\
             projection=projStr) # create contour fill. dpi controls smoothness
fig.grdcontour(grid=calcGrid,levels=2,annotation=10,
             frame=['ag5000','+tCalculated Gravity Map, Contrast='+ str(rho) + \
                    ', Masked='+str(maskRad)+' m, BlockMed='+str(useBlockMed),\
                            "xaf+lEasting (m)", "yaf+lNorthing (m)"]) # create contour lines. levels is contour interval. frame is border style and grid lines
fig.colorbar(frame=['a5f5','x+lGravity Anomaly (mGal)'],  # Set annotations by 5 mGal
             position="JRM") # Place colorbar at position Right Middle

fig.show()


if useBlockMed == True:
    saveFileName = 'calcGravMap_'+str(gridSpace)+'m'+str(rho)+'Mask'+str(maskRad)+'m_BlockMed.png'
else:
    saveFileName = 'calcGravMap_'+str(gridSpace)+'m'+str(rho)+'Mask'+str(maskRad)+'m.png'

fig.savefig(saveDir+saveFileName) # save figure


del(calcGrid,fig,saveFileName)


#%% 12: PyGMT Difference gravity map plotting

diffGrid = pg.xyz2grd(x=dfMasked.iloc[:,0],y=dfMasked.iloc[:,1],
             z=dfMasked.iloc[:,5], spacing=gridSpace, region=regionList)
    

fig = pg.Figure()

fig.grdimage(region=regionList, grid=diffGrid,cmap='haxby',dpi=300,\
             projection=projStr) # create contour fill. dpi controls smoothness
fig.grdcontour(grid=diffGrid,levels=1,annotation=2,
             frame=['ag5000','+tDifference Gravity Map, Contrast='+ str(rho) + \
                    ', Masked='+str(maskRad)+' m, BlockMed='+str(useBlockMed),\
                            "xaf+lEasting (m)", "yaf+lNorthing (m)"]) # create contour lines. levels is contour interval. frame is border style and grid lines
fig.colorbar(frame=['a2f2','x+lGravity Anomaly (mGal)'],  # Set annotations by 2 mGal
             position="JRM") # Place colorbar at position Right Middle

fig.show()


if useBlockMed == True:
    saveFileName = 'diffGravMap_'+str(gridSpace)+'m'+str(rho)+'Mask'+str(maskRad)+'m_BlockMed.png'
else:
    saveFileName = 'diffGravMap_'+str(gridSpace)+'m'+str(rho)+'Mask'+str(maskRad)+'m.png'

fig.savefig(saveDir+saveFileName) # save figure


del(diffGrid,fig,saveFileName)


#%% 13: PyGMT Slope map plotting

slopeAzGrid = pg.grdgradient(thickGrid,direction='a',slope_file=saveDir+'slopeMask.grd') #create xarray of slope azimuths and save magnitudes to file
slopeGrid = xr.open_dataarray(saveDir+'slopeMask.grd') # import slope magnitude file as xarray

slopeDegGrid = np.degrees(np.arctan(slopeGrid)) # convert slopes to degrees and creat xarray
slopeDegGrid.to_netcdf(saveDir+'slopeDegMask.grd') # save xarray to grd file


fig = pg.Figure()

fig.grdimage(region=regionList, grid=slopeDegGrid, cmap='dem2', dpi=300,\
             projection=projStr) # create contour fill. dpi controls smoothness
fig.grdcontour(grid=slopeDegGrid, levels=5, annotation=25,
             frame=['ag5000','+tBasement Slope Map, Contrast='+ str(rho) + \
                    ', Masked='+str(maskRad)+' m, BlockMed='+str(useBlockMed),\
                            "xaf+lEasting (m)", "yaf+lNorthing (m)"]) # create contour lines. levels is contour interval. frame is border style and grid lines
fig.colorbar(frame=['a10f10','x+lSlope (degrees)'],  # Set annotations by 10 degrees
             position="JRM") # Place colorbar at position Right Middle

fig.show()


if useBlockMed == True:
    saveFileName = 'slopeMap_'+str(gridSpace)+'m'+str(rho)+'Mask'+str(maskRad)+'m_BlockMed.png'
else:
    saveFileName = 'slopeMap_'+str(gridSpace)+'m'+str(rho)+'Mask'+str(maskRad)+'m.png'

fig.savefig(saveDir+saveFileName) # save figure


del(saveFileName)

