# -*- coding: utf-8 -*-
'''
Title: 2D Gravity Profile Modeling
Desc: Modeling basin gravity profiles from gravity data using a single density
    contrast. Uses a modified Plouff (1976) method with prisms pseudo-infinite
    in cross profile direction. Uses 4 interpolation methods for quick comparison.
    Coordinates must be in UTM (m) and gravity in mGal. Finds gravity survey points
    near a profile line given its endpoints and projects points onto the line.
Author: Bradford Mack
Date: 20 Mar 24
Last modified: 5 Nov 24
'''

from datetime import datetime as dt
import gravModelLib as gm
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
from tqdm import tqdm
import yaml


#%% Load inputs

paramfile = 'ProfileModelParams.txt'

with open(paramfile, 'r') as yml:
    params = yaml.safe_load(yml)


mainSaveDir = params['Main Save Directory'] # main project folder to save all data and figures. Code will create subfolders in this directory. Must have / at the end
dfProfile = pd.read_csv(mainSaveDir+params['Data File'],header=0) # Import grav data. Must have 4 columns: Easting (m), Northing (m), Elevation (m), and Bouguer Anomaly (mGal)
dfProfileCoord = pd.read_csv(mainSaveDir+params['Profile Coordinates File'],header=0) # Import profile coordinates. Must have 5 columns: Profile Number, Start X (m), Start Y (m), End X (m), End Y (m)
profileNum = params['Profile Number'] # profile number from Profile Coordinates File to plot
profileRad = params['Profile Search Radius'] # distance away from profile line to include gravity survey points

rho = params['Density Contrast'] # density contrast in kg/m^3
gridSpace = params['Grid Spacing'] # spacing of interpolation grid in meters
maxDepth = params['Maximum Depth'] # maximum basin depth in m
z0 = params['Station Depth'] # depth of gravity staion. 0 is at the surface
prismTop = params['Prism Top'] # depth of top of prism in m. Should not be 0.
prismLength = params['Prism Length'] # long axis of prism for modeling. Should be long enough to make it "infinite"

badPoints = params['Bad Point Indices'] # list of indices for points to be deleted.
detrendPts = params['Detrend Point Indices'] # indices of points that the detrend line will set to 0 mGal. Indices correspond to the indices in the dfProfile dataFrame just before running the interpolation in the code.


del(params,yml)


#%% Error handling

assert gridSpace > 0, 'Grid spacing must be > 0'
assert rho < 0, 'Density contrast must be < 0'
assert z0 <= 0, 'Depth of gravity station must be <= 0 (depth is positive down)'
assert maxDepth > 0, 'Max depth must be > 0'
assert prismTop > 0, 'Prism top depth must be > 0'
assert prismLength > 0, 'Prism top depth must be > 0'
assert profileRad >= 0, 'Mask radius must be >= 0'
if 'badPoints' in globals():
    assert type(badPoints) == list, 'Points for removal must be in a list'
if 'detrendPts' in globals():
    assert type(detrendPts) == list, 'Points for detrending must be in a list'
if 'detrendPts' in globals():
    assert len(detrendPts) == 1 or len(detrendPts) == 2, 'Must give only 1 or 2 points for detrending'    
    

#%% 2D profile data parsing

proSaveDir = mainSaveDir+'Figures/Profiles/' # directory to save profile data and figures in
figSaveDir = proSaveDir+'profile'+str(profileNum)+'/' # directory to save profile figures and data in
dataSaveDir = mainSaveDir+'ProfileData/' # subdirectory to save profile data in

if not os.path.isdir(proSaveDir): # create drirectory if it doesn't exist
    os.makedirs(proSaveDir)
if not os.path.isdir(figSaveDir): # create drirectory if it doesn't exist
    os.makedirs(figSaveDir)
if not os.path.isdir(dataSaveDir): # create drirectory if it doesn't exist
    os.makedirs(dataSaveDir)

dataSaveFileName = 'Profile'+str(profileNum)+'Data.csv' # root file name for all profile data
figSaveFileName = 'Profile'+str(profileNum)+'Data.png' # file name for profile data figure


dfProfile.insert(loc=4,column='Nearest Point Easting (m)',value=0.0) # add column for distance from profile line to use for parsing
dfProfile.insert(loc=5,column='Nearest Point Northing (m)',value=0.0) # add column for distance from profile line to use for parsing
dfProfile.insert(loc=6,column='Distance From Profile Line (m)',value=0.0) # add column for distance from profile line to use for parsing
dfProfile.insert(loc=7,column='Distance Along Profile (m)',value=0.0) # add column for distance of projected point along profile line
dfProfile.insert(loc=8,column='Inside Search Area',value=True) # add column for whether the point is within the profile search area


for i in range(len(dfProfileCoord)): # loop through profile coordinate dataframe to find profile coordinates
    if dfProfileCoord.iloc[i,0] == profileNum: # find profile data by number 
        sX = dfProfileCoord.iloc[i,1] # x coordinate of profile start
        sY = dfProfileCoord.iloc[i,2] # y coordinate of profile start
        eX = dfProfileCoord.iloc[i,3] # x coordinate of profile end
        eY = dfProfileCoord.iloc[i,4] # y coordinate of profile end        


for i in range(len(dfProfile)): # loop through all grav stations to find which are close to profile line
    lineData = gm.pointOnLine(sX, sY, eX, eY,dfProfile.iloc[i,0], dfProfile.iloc[i,1]) # calculate points along line nearest each station and distance between them
    dfProfile.iloc[i,4] = lineData[0] # store nearest point Easting
    dfProfile.iloc[i,5] = lineData[1] # store nearest point Northing
    dfProfile.iloc[i,6] = lineData[2] # store nearest point distance from profile
    dfProfile.iloc[i,7] = lineData[3] # store nearest point distance along profile

dfProfile = dfProfile[dfProfile.iloc[:,6]<=profileRad] # remove points outside of search distance (profileRad). The line is infinite in the distance calculation, however.


profileAz, sQuad, eQuad = gm.profileOrient(sX, sY, eX, eY) # find azimuth of profile and endpoint quadrants for labeling plots

polyCorners = gm.profileCorners(sX, sY, eX, eY, profileRad) # find corner coordinates for profile search area
       
polygon = Path([polyCorners[0],polyCorners[1],polyCorners[2],polyCorners[3],polyCorners[0]]) # create a path object using the search area corners. The 5th point is needed to close the polygon on the plot
polyPatch = PathPatch(polygon, facecolor='none', edgecolor='green') # create a polygon object for testing if points are inside the area


for i in range(len(dfProfile)): # loop through grav data to see if each point is within the search area. Necessary because gm.pointOnLine projects line infinitely
    dfProfile.iloc[i,8] = polygon.contains_point((dfProfile.iloc[i,0],dfProfile.iloc[i,1]))

dfProfile = dfProfile[dfProfile.iloc[:,8]==True] # drop all rows that are outside of the search area
dfProfile.sort_values(by=['Distance Along Profile (m)'], inplace=True, ignore_index=True) # sort by distance along profile
dfProfile.to_csv(dataSaveDir+dataSaveFileName[:-4]+'.csv', index=False)


del(i,lineData)


#%%% Trim bad points from profile data

if 'badPoints' in globals(): # check if any points marked for deletion
    dfProfile.drop(index=badPoints,inplace=True) # drop rows of bad points
    dfProfile.reset_index(drop=True,inplace=True) # reset dataframe indices after dropping rows


badPoints = [] # list for averaging two colocated points
for i in range(len(dfProfile)-1): # loop through all but last row
    for j in range(i+1,len(dfProfile)): # loop through all rows after i
        if dfProfile.iloc[i,0] == dfProfile.iloc[j,0] and dfProfile.iloc[i,1] == dfProfile.iloc[j,1]: # check for two identical locations
            dfProfile.iloc[i,3] = (dfProfile.iloc[i,3]+dfProfile.iloc[j,3])/2 # average gravity of two colocated points
            badPoints.append(j) # add index of second matched point to list for deletion

if len(badPoints) > 0: # check if any points marked for deletion
    dfProfile.drop(index=badPoints,inplace=True) # drop rows of bad points
    dfProfile.reset_index(drop=True,inplace=True) # reset dataframe indices after dropping rows


fig = plt.figure() # new figure
ax = fig.add_subplot() # add plot to fig
ax.plot([sX,eX],[sY,eY]) # plot profile line
ax.scatter(dfProfile.iloc[:,0],dfProfile.iloc[:,1],color='orange') # plot grav survey points
fig.set_figwidth(16)
ax.set_aspect(1) # set aspect ratio to 1
ax.add_patch(polyPatch) # plot search area rectangle

plt.suptitle('Profile '+str(profileNum)+' Data')

fig.savefig(figSaveDir+figSaveFileName)


del(ax,badPoints,eX,eY,i,j,polyCorners,polygon,polyPatch,profileRad,sX,sY)


#%% Plot gravity and elevation profiles

profileXint = gm.roundDown(max(dfProfile.iloc[:,7])/10, 500) # set x axis intervals
#  to have at least 10 and rounded to a multiple of 500. Comment out for auto 
# labels or change to any interval you want.

fig, axs = plt.subplots(2,sharex=True) #create subplots

plt.axes(axs[0]) # gravity subplot
plt.plot(dfProfile.iloc[:,7],dfProfile.iloc[:,3],'-o')
axs[0].set_ylabel('Gravity (mGal)')
axs[0].set_title(sQuad, loc='left')
axs[0].set_title(eQuad, loc='right')

plt.axes(axs[1]) # elevation subplot
plt.plot(dfProfile.iloc[:,7],dfProfile.iloc[:,2],'-o')
axs[1].set_ylabel('Elevation (m)')
axs[1].set_ylim(1400, 2400)


axs[1].set_xlabel('Horizontal Distance (m)') # x label at bottom
if 'profileXint' in globals():
    plt.xticks(np.arange(0, max(dfProfile.iloc[:,7])+profileXint, profileXint)) # create x ticks/labels at set interval

plt.suptitle('Gravity and Elevation Profile ' + str(profileNum))

fig.set_figwidth(16)
fig.set_figheight(12)

figSaveFileName = 'GravElevProfile'+str(profileNum)+'.png' # file name for grav/elevation figure

fig.savefig(figSaveDir+figSaveFileName)


#%% Linear detrend profile data

# This will only run if you input a list of points to use for a linear detrend.

if 'detrendPts' in globals(): 
    dfProfileDT = dfProfile.copy()
    
    if len(detrendPts) == 2:
        slope = (dfProfile.iloc[detrendPts[1],3]-dfProfile.iloc[detrendPts[0],3])/\
            (dfProfile.iloc[detrendPts[1],7]-dfProfile.iloc[detrendPts[0],7]) # slope of detrend line
        yInt = dfProfile.iloc[detrendPts[0],3]-slope*dfProfile.iloc[detrendPts[0],7] # y intercept of detrend line
    
    elif len(detrendPts) == 1:
        slope = 0 # set slope to 0 to shift all points the same amount
        yInt = dfProfile.iloc[detrendPts[0],3] # set y intercept to y value of detrend point
        
    
    for i in range(len(dfProfile)): # loop through all gravity points
        dfProfileDT.iloc[i,3] -= slope*dfProfile.iloc[i,7]+yInt # remove linear trend from gravity data
    
    dfProfileDT.to_csv(dataSaveDir+dataSaveFileName[:-4]+'_DT.csv', index=False) # save detrended data to CSV
    
    
    fig, axs = plt.subplots(2,sharex=True) #create subplots
    
    plt.axes(axs[0])
    plt.plot(dfProfile.iloc[:,7],dfProfile.iloc[:,3],'-o')
    axs[0].set_ylabel('Gravity (mGal)')
    axs[0].set_title(sQuad, loc='left')
    axs[0].set_title(eQuad, loc='right')
    
    plt.axes(axs[1])
    plt.plot(dfProfileDT.iloc[:,7],dfProfileDT.iloc[:,3],'-o')
    axs[1].set_ylabel('Detrended Gravity (mGal)')
    
    
    axs[1].set_xlabel('Horizontal Distance (m)') # x label at bottom
    if 'profileXint' in globals():
        plt.xticks(np.arange(0, max(dfProfile.iloc[:,7])+profileXint, profileXint))
    
    plt.suptitle('Detrended Gravity Profile ' + str(profileNum))
    
    fig.set_figwidth(16)
    fig.set_figheight(12)

    figSaveFileName = 'GravProfile'+str(profileNum)+'_DT.png' # file name for detrended grav figure
    
    fig.savefig(figSaveDir+figSaveFileName)
    
    
    del(detrendPts,i,slope,yInt)


#%% Gravity interpolation

# This should not be done without detrending your data, so it requires the dataframe 
# produced above. After running your first model, choose your detrend points, 
# input them at the top, and run it again.
    
if 'dfProfileDT' in globals(): 
    
    y0 = 0 # y coordinate (horizontal and perpendicular to profile line) of gravity station
    methodList = ['linear','akima','pchip','spline']
    
    for method in methodList:
        
        print('Interpolating '+method)
        
        xInt = np.arange(0,dfProfileDT.iloc[-1,7],gridSpace) # create array of x values for interpolation
        
        if method == 'linear':
            gravInt = np.interp(xInt,dfProfileDT.iloc[:,7],dfProfileDT.iloc[:,3]) # linear interpolation of gravity data
        
        elif method == 'akima':
            gravInt = Akima1DInterpolator(dfProfileDT.iloc[:,7],dfProfileDT.iloc[:,3])(xInt) # interpolation using Akima1DInterpolator
        
        elif method == 'pchip':
            gravInt = PchipInterpolator(dfProfileDT.iloc[:,7],dfProfileDT.iloc[:,3])(xInt) # interpolation using PchipInterpolator
            
        elif method == 'spline':
            gravInt = CubicSpline(dfProfileDT.iloc[:,7],dfProfileDT.iloc[:,3])(xInt) # interpolation using CubicSpline
        
        data = np.vstack((xInt,gravInt)).T # create 2D numpy array of interpolated distance and gravity values
        dfProfileInt = pd.DataFrame(data,columns=['Profile Distance (m)','Interpolated Gravity (mGal)']) # create dataframe of interpolated data
        
        dfProfileInt.to_csv(dataSaveDir+dataSaveFileName[:-4]+'_Int_'+method+'.csv', index=False) # save interpolated profile data to csv
        
        
        fig, axs = plt.subplots(2,sharex=True) #create subplots
    
        plt.axes(axs[0])
        plt.plot(dfProfileDT.iloc[:,7],dfProfileDT.iloc[:,3],'-o')
        axs[0].set_ylabel('Detrended Gravity (mGal)')
        axs[0].set_title(sQuad, loc='left')
        axs[0].set_title(eQuad, loc='right')
     
        plt.axes(axs[1])
        plt.plot(dfProfileDT.iloc[:,7],dfProfileDT.iloc[:,3],'o')
        plt.plot(dfProfileInt.iloc[:,0],dfProfileInt.iloc[:,1],'-o')
        axs[1].set_ylabel('Interpolated Gravity (mGal)')
    
        
        axs[1].set_xlabel('Horizontal Distance (m)') # x label at bottom
        if 'profileXint' in globals():
            plt.xticks(np.arange(0, max(dfProfile.iloc[:,7])+profileXint, profileXint))
        plt.suptitle('Interpolated Gravity Profile - ' + method)
        
        fig.set_figwidth(16)
        fig.set_figheight(12)        
        
        figSaveFileName = 'GravProfile'+str(profileNum)+'_Int_'+method+'.png' # file name for interpolated grav figure

        fig.savefig(figSaveDir+figSaveFileName)
        
        
        del(data)
            
        
        #% Profile modeling
        
        y1 = y0-prismLength # y coordinate of left edge of prism
        y2 = y0+prismLength # y coordinate of right edge of prism
        
        dfProfileInt.dropna(inplace=True,ignore_index=True) # remove rows with NaN values in gravity column. Akima method does not interpolate at 0 m
        
        if len(dfProfileInt.columns) == 2: # add columns if necessary
            dfProfileInt.insert(loc=2,column='Thickness (m)',value=0.0) # add column to store basin thickness values
            dfProfileInt.insert(loc=3,column='Calc Grav (mGal)',value=0.0) # add column to store calculated gravity values
            dfProfileInt.insert(loc=4,column='Diff Grav (mGal)',value=0.0) # add column to store difference between observed and calculated gravity values
            dfProfileInt.insert(loc=5,column='d Thick (m)',value=0.0) # add column to store change in thickenss due to gravity difference values
        
                
        dfPrism = pd.DataFrame(np.zeros((len(dfProfileInt),1),dtype=float),columns=['X Boundary 1 (m)'])
        dfPrism.insert(loc=1,column='Y Boundary 1 (m)',value=y1)
        dfPrism.insert(loc=2,column='Prism Top (m)',value=prismTop)
        dfPrism.insert(loc=3,column='X Boundary 2 (m)',value=0.0)
        dfPrism.insert(loc=4,column='Y bounday 2 (m)',value=y2)
        dfPrism.insert(loc=5,column='Thickness (m)',value=0.0)
        dfPrism.insert(loc=6,column='Density Contrast (kg/m^3)',value=rho)
        
        
        for i in range(len(dfProfileInt)): # calculate initial basin depth using simple Bouguer
            if dfProfileInt.iloc[i,1] < 0:
                dfPrism.iloc[i,5] = gm.bouguer(dfProfileInt.iloc[i,1],rho) # calculate depth of basin in m
                dfProfileInt.iloc[i,2] = dfPrism.iloc[i,5]
            else:
                dfPrism.iloc[i,5] = 0 # set basin depth to 0 m
                dfProfileInt.iloc[i,2] = 0
        
        for i in range(len(dfPrism)): # fill prism dataframe
            dfPrism.iloc[i,0] = dfProfileInt.iloc[i,0]-gridSpace/2
            dfPrism.iloc[i,3] = dfProfileInt.iloc[i,0]+gridSpace/2
            
        dfProfileInt.to_csv(dataSaveDir+dataSaveFileName[:-4]+'_Model_'+method+'_Initial.csv', index=False) # save inital data to csv
        
        
            
        for step in range(10): # run prism calculation multiple times and adjust thickness to minimize obs-calc error
            for i in tqdm(range(len(dfProfileInt))): # iterate through all observation points on the grid
                dfProfileInt.iloc[i,3] = 0 # zero out calculated gravity before running next iteration
                for j in range(len(dfPrism)): # iterate over every prism and sum gravity at OP
                    dfProfileInt.iloc[i,3] += gm.gravPrism(dfProfileInt.iloc[i,0],\
                            y0,z0,dfPrism.iloc[j,0],dfPrism.iloc[j,1],\
                            dfPrism.iloc[j,2],dfPrism.iloc[j,3],dfPrism.iloc[j,4],\
                            dfPrism.iloc[j,5],dfPrism.iloc[j,6])  
              
                dfProfileInt.iloc[i,4] = dfProfileInt.iloc[i,1]-dfProfileInt.iloc[i,3] # find the difference between calculated and observed gravity
                     
                dThick = gm.bouguer(dfProfileInt.iloc[i,4],rho) # calculate difference in thickness due to difference in gravity
                dfProfileInt.iloc[i,5] = dThick
                dfProfileInt.iloc[i,2] += dThick #adjust thickness for final maps
                dfPrism.iloc[i,5] += dThick #adjust thickness for next iteration of prism calculations
            
                if dfProfileInt.iloc[i,2] < 0: # do not allow negative thickness
                    dfProfileInt.iloc[i,2] = 0
                if dfProfileInt.iloc[i,2] > maxDepth: # do not exceed max basin depth
                    dfProfileInt.iloc[i,2] = maxDepth
                if dfPrism.iloc[i,5] < 0: # do not allow negative thickness
                    dfPrism.iloc[i,5] = 0
                if dfPrism.iloc[i,5] > maxDepth: # do not exceed max basin depth
                    dfPrism.iloc[i,5] = maxDepth
                    
                #print(str(i+1)+' of '+str(len(dfProfileInt))+' OPs complete')
            print('Step ',step,' complete. '+ str(dt.now().time()))
            
            dfProfileInt.to_csv(dataSaveDir+dataSaveFileName[:-4]+'_Model_'+method+'_Step'+str(step)+'.csv', index=False) # save dataframe to csv
            
            
            
            
        fig, axs = plt.subplots(2,sharex=True) #create subplots
        
        plt.axes(axs[0])
        plt.plot(dfProfileDT.iloc[:,7],dfProfileDT.iloc[:,3],'o')
        plt.plot(dfProfileInt.iloc[:,0],dfProfileInt.iloc[:,1],'-o')
        axs[0].set_ylabel('Interpolated Gravity (mGal)')
        axs[0].set_title(sQuad, loc='left')
        axs[0].set_title(eQuad, loc='right')
    
        
        plt.axes(axs[1])
        plt.plot(dfProfileInt.iloc[2:,0],dfProfileInt.iloc[2:,2],'-o')
        plt.gca().invert_yaxis()
        axs[1].set_ylabel('Basin Depth (m)')
    
        
        axs[1].set_xlabel('Horizontal Distance (m)') # x label at bottom
        if 'profileXint' in globals():
            plt.xticks(np.arange(0, max(dfProfile.iloc[:,7])+profileXint, profileXint))
        plt.suptitle('Modeled Basin Depth - ' + method)
        
        fig.set_figwidth(16)
        fig.set_figheight(12)
       
        figSaveFileName = 'GravProfile'+str(profileNum)+'_Depth_'+method+'.png' # file name for interpolated grav figure
         
        fig.savefig(figSaveDir+figSaveFileName)
            
        
    del(dThick,gravInt,gridSpace,i,j,maxDepth,method,methodList,prismLength,prismTop,rho,step,\
        xInt,y0,y1,y2,z0)
    

#%% Fault estimation

# I used this section to estimate the dip of a basin bounding fault. This is a niche
# use, so this is not meant to be run with all models. I left it in just in case
# it is useful for others. You will need to choose your fault end points as I have
# done below. You can also use it to estimate the slope between any two points
# but it will label them as "fault". 

'''
#method = 'linear' # method of interpolation. Choose linear, akima, pchip, or spline
#method = 'akima'
method = 'pchip'
#method = 'spline'

readFile = dataSaveDir+dataSaveFileName[:-4]+'_DT.csv'
dfProfileDT = pd.read_csv(readFile,header=0)

readFile = dataSaveDir+dataSaveFileName[:-4]+'_Model_'+method+'_Step'+str(9)+'.csv'
dfProfileInt = pd.read_csv(readFile,header=0)


faultStart = [dfProfileInt.iloc[-6,0],dfProfileInt.iloc[-6,2]] # starting coordinates of fault to draw (x,y)
faultEnd = [dfProfileInt.iloc[-2,0],dfProfileInt.iloc[-2,2]] # ending coordinates of fault to draw (x,y)
faultDip = np.degrees(abs(np.arctan((faultEnd[1]-faultStart[1])/\
                                    (faultEnd[0]-faultStart[0])))) # dip of fault in degrees
faultMid = [np.average((faultStart[0],faultEnd[0])),np.average((faultStart[1],faultEnd[1]))] # midpoint of fault line (x,y) for plotting text
textOff = (faultEnd[0]-faultStart[0])*0.2 # x offset for text plotting

    


fig, axs = plt.subplots(2,sharex=True) #create subplots

plt.axes(axs[0])
plt.plot(dfProfileDT.iloc[:,7],dfProfileDT.iloc[:,3],'o') # plot detrended profile data
plt.plot(dfProfileInt.iloc[:,0],dfProfileInt.iloc[:,1],'-o')
axs[0].set_ylabel('Interpolated Gravity (mGal)')
axs[0].set_title(sQuad, loc='left')
axs[0].set_title(eQuad, loc='right')


plt.axes(axs[1])
plt.plot(dfProfileInt.iloc[:,0],dfProfileInt.iloc[:,2],'-o')
plt.plot([faultStart[0],faultEnd[0]],[faultStart[1],faultEnd[1]],'-')
plt.gca().invert_yaxis()
axs[1].set_ylabel('Basin Depth (m)')


textOff = max(dfProfileInt.iloc[:,0])/12

axs[1].annotate('{:2.0f}\u00b0 fault dip'.format(faultDip), xy=(faultMid[0],faultMid[1]),\
            xytext=(faultMid[0]+textOff,faultMid[1]),\
            arrowprops=dict(facecolor='black', shrink=0.02),)


axs[1].set_xlabel('Horizontal Distance (m)') # x label at bottom
if 'profileXint' in globals():
    plt.xticks(np.arange(0, max(dfProfile.iloc[:,7])+profileXint, profileXint))
plt.suptitle('Modeled Basin Depth - ' + method)
   

figSaveFileName = 'GravProfile'+str(profileNum)+'_NoUTEP_Depth_'+method+'_fault.png' # file name for interpolated grav figure
fig.savefig(figSaveDir+figSaveFileName)

'''

