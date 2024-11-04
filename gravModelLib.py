# -*- coding: utf-8 -*-
'''
Title: Gravity Modeling Library
Desc: Library of functions used for gravity modeling. Functions for 2D and 3D basin models.
Author: Bradford Mack
Date: 20 Mar 24
Last modified: 29 May 24
'''

def bouguer(grav,rho):
    '''
    Calculate the thickness of a body from a simple Bouguer anomaly and an assumed
    density contrast for a single point.
    
    Parameters
    ----------
    grav : int or float
        Gravity anomaly in mGal.
    rho : int or float
        Density contrast in kg/m^3.

    Returns
    -------
    thickness : float
        Calculated thickness in meters.

    '''
    
    
    from math import pi
    
    gConstant = 6.67e-11 # gravitational constant in Nm^2/kg^2
    twoPi = 2*pi # two times pi to avoid repeated calculation of it
    
    thickness = grav/(twoPi*gConstant*rho)/100000 # Calculate thickness in m. 100000 is to convert from mGal to SI

    return thickness



def bouguerDF(dfGravDataGrid,rho,gravColumn=2,thickColumn=3):
    '''
    Calculate the thickness of a body from a simple Bouguer anomaly and an assumed
    density contrast. Must be passed a dataframe.
    
    Parameters
    ----------
    dfGravDataGrid : Pandas dataframe
        Dataframe containing gravity anomaly data in mGal.
    rho : int or float
        Density contrast in kg/m^3.
    gravColumn : int, optional
        The default is 2. Column number containing gravity anomaly values.
    thickColumn : int, optional
        The default is 3. Column number for output thickness values.
        If no thickness column already exists, it will create one at the given index.
        Must be no greater than the width of the dataframe (i.e. if df dimensions 
        are 20,3 then a max index of 3 can be used which will create a new column).

    Returns
    -------
    dfGravDataGrid : Pandas dataframe
        Dataframe containing all input data plus thickness values calculated by function.

    '''
    
    
    from math import pi
    
    gConstant = 6.67e-11 # gravitational constant in Nm^2/kg^2
    twoPi = 2*pi # two times pi to avoid repeated calculation of it
    
    if len(dfGravDataGrid.columns) == thickColumn:
        dfGravDataGrid.insert(loc=thickColumn,column='Thickness (m)',value=0.0) # add column to store basin thickness values
    
    for i in range(len(dfGravDataGrid)): # calculate thickness from simple Bouguer gravity anomaly
        if dfGravDataGrid.iloc[i,gravColumn] < 0: # only calculate when gravity anomaly is negative
            dfGravDataGrid.iloc[i,thickColumn] = dfGravDataGrid.iloc[i,gravColumn]/(twoPi*gConstant*rho)/100000 # Calculate thickness in m. 100000 is to convert from mGal to SI

    return dfGravDataGrid



def gravPrism(x0,y0,z0,x1,y1,z1,x2,y2,z2,rho):
    '''
    Calculates the vertical attraction of a rectangular prism given a density 
    contrast and coordinates of observation point and prism edges. Sides of prism
    are parallel to x, y, and z axes. Coordinates must in meters. X is positive
    east. Y is positive north. Z is positive down. Based on Plouff method (1976).

    Parameters
    ----------
    x0 : int or float
        X coordinate of observation point in m.
    y0 : int or float
        Y coordinate of observation point in m.
    z0 : int or float
        Z coordinate (depth) of observation point in m.
    x1 : int or float
        Western boundary coordinate of prism in m.
    y1 : int or float
        Southern boundary coordinate of prism in m.
    z1 : int or float
        Depth of top of prism in m.
    x2 : int or float
        Eastern boundary coordinate of prism in m.
    y2 : int or float
        Northern boundary coordinate of prism in m.
    z2 : int or float
        Depth of bottom of prism in m.
    rho : int or float
        Density contrast between prism and surrounding rock in kg/m^3.

    Returns
    -------
    g : float
        Calculated vertical attraction of gravity in mGal.

    '''

    from math import atan2
    from math import log
    from math import pi
    from numpy import sqrt
    
    notZero = 1e-10 # value to prevent divide by 0 errors
    gConstant = 6.67e-11 # gravitational constant in Nm^2/kg^2
    twoPi = 2*pi # two times pi to avoid repeated calculation of it
    si2mGal = 100000 # conversion factor to go from SI to mGal (*) or reverse (/)
    
    #z1 = 0.1 # depth of top of prism in m
    #z2 = 100 # depth of bottom of prism (thickness)
    
    xs = [] # list of x differences between x0 and prism edges
    ys = [] # list of y differences between y0 and prism edges
    zs = [] # list of z differences between z0 and prism edges
    isign = [] # list of -1 and 1 to change signs in calculation
    
    xs.append(x0-x1) # x difference between x0 and west edge of prism
    ys.append(y0-y1) # y difference between y0 and south edge of prism
    zs.append(z0-z1) # z difference between z0 and top edge of prism
    xs.append(x0-x2) # x difference between x0 and east edge of prism
    ys.append(y0-y2) # y difference between y0 and north edge of prism
    zs.append(z0-z2) # z difference between z0 and bottom edge of prism
    isign.append(-1)
    isign.append(1)
    
    gSum=0
    for x in range(2):
        for y in range(2):
            for z in range(2):
                rijk = sqrt(xs[x]**2 + ys[y]**2 + zs[z]**2) # distance from obs point to prism corner
                ijk = isign[x]*isign[y]*isign[z];
                arg1 = atan2((xs[x]*ys[y]),(zs[z]*rijk))
                if arg1 < 0: 
                    arg1 = arg1 + twoPi
                arg2 = rijk+ys[y]
                arg3 = rijk+xs[x]
                if arg2 <= 0:
                    arg2 = notZero
                if arg3 <= 0.0:
                    arg3 = notZero
                arg2 = log(arg2)
                arg3 = log(arg3)
                gSum += ijk*(zs[z]*arg1-xs[x]*arg2-ys[y]*arg3)
    
    
    g = rho*gSum*gConstant*si2mGal
    
    return g



def roundDown(x,r):
    '''
    Rounds x down to the nearest given value/interval (r). For example, to round
    down to the nearest thousand, r=1000.

    Parameters
    ----------
    x : int or float
        Value to be rounded.
    r : int or float
        Value/interval to be rounded down to. Can be any number and the function
        will round down to the nearest number evenly divisble by r.

    Returns
    -------
    int or float
        Value of x rounded down to the nearest multiple of r.

    '''
    
    import math
    
    return int(math.floor(x/r))*r



def roundUp(x,r):
    '''
    Rounds x up to the nearest given value/interval (r). For example, to round
    up to the nearest thousand, r=1000.

    Parameters
    ----------
    x : int or float
        Value to be rounded.
    r : int or float
        Value/interval to be rounded up to. Can be any number and the function
        will round up to the nearest number evenly divisble by r.

    Returns
    -------
    int or float
        Value of x rounded up to the nearest multiple of r.

    '''
    
    import math
    
    return int(math.ceil(x/r))*r



def pointOnLine(startX,startY,endX,endY,testPointX,testPointY):
    '''
    Calculates the point along a line nearest another point given the line 
    endpoint coordinates and test point coordinates. Only works on a cartesian
    coordinate system.

    Parameters
    ----------
    startX : int or float
        X coordinate of the start of the line.
    startY : int or float
        Y coordinate of the start of the line.
    endX : int or float
        X coordinate of the end of the line.
    endY : int or float
        Y coordinate of the end of the line.
    testPointX : int or float
        X coordinate of the test/reference point to find the point nearest along the line
    testPointY : int or float
        Y coordinate of the test/reference point to find the point nearest along the line


    Returns
    -------
    lineX : float
        X coordinate of point along line nearest to the test point
    lineY : float
        Y coordinate of point along line nearest to the test point
    distToLine : float
        Distance between test point and the nearest point along the line in whatever
        units the coordinates are in.
    profileDist : float
        Distance along profile line from start to nearest point projected onto profile line

    '''
    
    from numpy import sqrt

    if startX == endX: # line runs N-S
        lineX = startX # Easting does not change along line
        lineY = testPointY # Northing equal to testPoint Northing
    elif startY == endY: # line runs E-W
        lineX = testPointX # Easting equal to testPoint Easting
        lineY = startY # Northing does not change along line
    else:
        slope = (endY-startY)/(endX-startX) # slope of line
        yInt = startY-slope*startX # y intercept of line
        lineX = (-(-testPointX-slope*testPointY)-slope*yInt)/(slope**2+1) # x coordinate of nearest point on line
        lineY = (slope*(testPointX+slope*testPointY)+yInt)/(slope**2+1) # y coordinate of nearest point on line

    distToLine = sqrt((testPointX-lineX)**2+(testPointY-lineY)**2) # calculate distance between test point and line
    profileDist = sqrt((startX-lineX)**2+(startY-lineY)**2) # calculate distance

    return lineX, lineY, distToLine, profileDist



def profileCorners(startX,startY,endX,endY,r):
    '''
    Calculates the coordinates of the corners of a rectangle centered on a line with
    a given radius (half-width). Returns coordinate pairs as tuples in the format (x,y).    
    Only works on a cartesian coordinate system. Points are numbered with pt 1 to the 
    left of the start point when facing the end point and moving clockwise from there.

    Parameters
    ----------
    startX : int or float
        X coordinate of start point of line.
    startY : int or float
        Y coordinate of start point of line.
    endX : int or float
        X coordinate of end point of line.
    endY : int or float
        Y coordinate of end point of line.
    r : int or float
        Radius/half-width of rectangle.

    Returns
    -------
    pt1x : float
        X coordinate of 1st corner
    pt1y : float
        Y coordinate of 1st corner
    pt2x : float
        X coordinate of 2nd corner
    pt2y : float
        Y coordinate of 2nd corner
    pt3x : float
        X coordinate of 3rd corner
    pt3y : float
        Y coordinate of 3rd corner
    pt4x : float
        X coordinate of 4th corner
    pt4y : float
        Y coordinate of 4th corner

    '''
    
    
    from math import cos
    from math import sin
    from math import atan

    if abs(endY-startY) != 0: # prevent divide by 0 error when line is E-W
        theta = atan(abs(endX-startX)/abs(endY-startY)) # angle of line relative to N


    if endX > startX and endY > startY: # line points NE
        pt1x = startX-r*cos(theta) # x coordinate of 1st corner
        pt1y = startY+r*sin(theta) # y coordinate of 1st corner
        pt2x = endX-r*cos(theta) # x coordinate of 2nd corner
        pt2y = endY+r*sin(theta) # y coordinate of 2nd corner
        pt3x = endX+r*cos(theta) # x coordinate of 3rd corner
        pt3y = endY-r*sin(theta) # y coordinate of 3rd corner
        pt4x = startX+r*cos(theta) # x coordinate of 4th corner
        pt4y = startY-r*sin(theta) # y coordinate of 4th corner

    elif endX > startX and endY < startY: # line points SE
        pt1x = startX+r*cos(theta) # x coordinate of 1st corner
        pt1y = startY+r*sin(theta) # y coordinate of 1st corner
        pt2x = endX+r*cos(theta) # x coordinate of 2nd corner
        pt2y = endY+r*sin(theta) # y coordinate of 2nd corner
        pt3x = endX-r*cos(theta) # x coordinate of 3rd corner
        pt3y = endY-r*sin(theta) # y coordinate of 3rd corner
        pt4x = startX-r*cos(theta) # x coordinate of 4th corner
        pt4y = startY-r*sin(theta) # y coordinate of 4th corner
        
    elif endX < startX and endY < startY: # line points SW
        pt1x = startX+r*cos(theta) # x coordinate of 1st corner
        pt1y = startY-r*sin(theta) # y coordinate of 1st corner
        pt2x = endX+r*cos(theta) # x coordinate of 2nd corner
        pt2y = endY-r*sin(theta) # y coordinate of 2nd corner
        pt3x = endX-r*cos(theta) # x coordinate of 3rd corner
        pt3y = endY+r*sin(theta) # y coordinate of 3rd corner
        pt4x = startX-r*cos(theta) # x coordinate of 4th corner
        pt4y = startY+r*sin(theta) # y coordinate of 4th corner

    elif endX < startX and endY > startY: # line points NW
        pt1x = startX-r*cos(theta) # x coordinate of 1st corner
        pt1y = startY-r*sin(theta) # y coordinate of 1st corner
        pt2x = endX-r*cos(theta) # x coordinate of 2nd corner
        pt2y = endY-r*sin(theta) # y coordinate of 2nd corner
        pt3x = endX+r*cos(theta) # x coordinate of 3rd corner
        pt3y = endY+r*sin(theta) # y coordinate of 3rd corner
        pt4x = startX+r*cos(theta) # x coordinate of 4th corner
        pt4y = startY+r*sin(theta) # y coordinate of 4th corner
        
    elif endX == startX: # line points N-S
        pt1x = startX-r # x coordinate of 1st corner
        pt1y = startY # y coordinate of 1st corner
        pt2x = endX-r # x coordinate of 2nd corner
        pt2y = endY # y coordinate of 2nd corner
        pt3x = endX+r # x coordinate of 3rd corner
        pt3y = endY # y coordinate of 3rd corner
        pt4x = startX+r # x coordinate of 4th corner
        pt4y = startY # y coordinate of 4th corner
        
    elif endY == startY: # line points E-W
        pt1x = startX # x coordinate of 1st corner
        pt1y = startY+r # y coordinate of 1st corner
        pt2x = endX # x coordinate of 2nd corner
        pt2y = endY+r # y coordinate of 2nd corner
        pt3x = endX # x coordinate of 3rd corner
        pt3y = endY-r # y coordinate of 3rd corner
        pt4x = startX # x coordinate of 4th corner
        pt4y = startY-r # y coordinate of 4th corner
        
    else:
        print('How did you manage this? Check profile coordinates.')
        
    return (pt1x, pt1y), (pt2x, pt2y), (pt3x, pt3y), (pt4x, pt4y)
        


def profileOrient(startX,startY,endX,endY):
    '''
    Calculates the azimuth of a profile line and which quadrants the endpoints
    lie in given cartesian coordinate pairs. Useful for labeling profile plots
    with their orientation. Azimuth is only valid if the Y coordinate is a Northing.

    Parameters
    ----------
    startX : int or float
        X coordinate of start point of line.
    startY : int or float
        Y coordinate of start point of line.
    endX : int or float
        X coordinate of end point of line.
    endY : int or float
        Y coordinate of end point of line.

    Returns
    -------
    profileAz : float
        0-360 azimuth of the profile line from the start toward the end.
    sQuad : string
        Quadrant that the start of the line points toward (N, NE, etc.)
    eQuad : TYPE
        Quadrant that the end of the line points toward (N, NE, etc.)

    '''
    
    import numpy as np

    relAz = abs(np.degrees(np.arctan((endX-startX)/(endY-startY)))) # angle of profile line relative to N-S
    
    # Find azimuth of profile line
    if startX < endX and startY < endY: 
        profileAz = relAz # azimuth of profile line
    elif startX < endX and startY > endY:
        profileAz = 180-relAz
    elif startX > endX and startY < endY:
        profileAz = 360-relAz
    elif startX > endX and startY > endY:
        profileAz = 180+relAz
    elif startX == endX and startY < endY: 
        profileAz = 0
    elif startX == endX and startY > endY:
        profileAz = 180      
    elif startX < endX and startY == endY:
        profileAz = 90
    elif startX > endX and startY == endY:   
        profileAz = 270
    
    # Find quadrants of profile ends for labeling plots
    if profileAz >= 337.5 or profileAz < 22.5:
        sQuad = 'S' # start point quadrant
        eQuad = 'N' # end point quadrant
    elif profileAz >= 22.5 and profileAz < 67.5:
        sQuad = 'SW'
        eQuad = 'NE'    
    elif profileAz >= 67.5 and profileAz < 112.5:
        sQuad = 'W'
        eQuad = 'E'    
    elif profileAz >= 112.5 and profileAz < 157.5:
        sQuad = 'NW'
        eQuad = 'SE'
    elif profileAz >= 157.5 and profileAz < 202.5:
        sQuad = 'N'
        eQuad = 'S'    
    elif profileAz >= 202.5 and profileAz < 247.5:
        sQuad = 'NE'
        eQuad = 'SW'    
    elif profileAz >= 247.5 and profileAz < 292.5:
        sQuad = 'E'
        eQuad = 'W'    
    elif profileAz >= 292.5 and profileAz < 337.5:
        sQuad = 'SE'
        eQuad = 'NW'   
        
    return profileAz, sQuad, eQuad
        