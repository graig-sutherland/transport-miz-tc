import numpy as np
from geopy.distance import vincenty
import seawater as sw
import math
from utils import meridional_curvature, prime_curvature

def get_distance(lon, lat, lo, la, phi):
    N, M = prime_curvature(phi)*np.cos(phi), meridional_curvature(phi)
    dlon, dlat = np.deg2rad(lon-lo), np.deg2rad(lat-la)
    dx, dy = N*dlon, M*dlat
    return dx, dy

def calculate_skill_score(dataObs, dataMod, method='molcard'):
    '''quick method to calculate different skill scores'''
    if len(dataMod) < len(dataObs):
        print('Mistmatch in calc skill')
        lon_mod = np.zeros_like(dataObs.Longitude.values)
        lat_mod = np.zeros_like(lon_mod)
        lon_mod[0], lat_mod[0] = dataObs.Longitude.values[0], dataObs.Latitude.values[0]
        lon_mod[1:], lat_mod[1:] = dataMod.Longitude.values, dataMod.Latitude.values
    else:
        lon_mod, lat_mod = dataMod.Longitude.values, dataMod.Latitude.values
    lon_obs, lat_obs = dataObs.Longitude.values, dataObs.Latitude.values
    tim_obs = (dataObs.index - dataObs.index[0]).values.astype('float')/1e9/3600.0
    phi = np.mean(np.deg2rad(dataObs.Latitude))
    dx, dy = get_distance(lon_mod[1:], lat_mod[1:], lon_obs[1:], lat_obs[1:], phi)
    sep = np.sqrt(dx**2 + dy**2)
    # need to calculate dxdt using time
    dxdt, dydt = np.gradient(dx, tim_obs[1:]), np.gradient(dy, tim_obs[1:])
    dsdt = np.sqrt(dxdt**2 + dydt**2)
    dxo, dyo = get_distance(lon_obs[1:], lat_obs[1:], lon_obs[:-1], lat_obs[:-1], phi)
    dist = np.sqrt(dxo**2 + dyo**2)
    dx0, dy0 = get_distance(lon_obs[1:], lat_obs[1:], lon_obs[0], lat_obs[0], phi)
    disp = np.sqrt(dx0**2 + dy0**2)
    dx0dt, dy0dt = np.gradient(dx0, tim_obs[1:]), np.gradient(dy0, tim_obs[1:])
    rel_dispers = np.sqrt( (dx*dxdt)**2 + (dy*dydt)**2 )
    dispers = np.sqrt( (dx0*dx0dt)**2 + (dy0*dy0dt)**2 )
    if method == 'molcard':
        skill = 1.0 - sep/disp
        # check limits and return skill
        skill[skill<0] = 0
        skill[skill>1] = 1
    elif method == 'toner':
        skill = 1.0 - sep / np.cumsum(dist)
        # check limits and return skill
        skill[skill<0] = 0
        skill[skill>1] = 1
    elif method == 'liu':
        skill = 1.0 - np.cumsum(sep) / np.cumsum(np.cumsum(dist))
        # check limits and return skill
        skill[skill<0] = 0
        skill[skill>1] = 1
    elif method == 'area':
        skill = 1.0 - np.cumsum(sep*dist) / np.cumsum(disp*dist)
        # check limits and return skill
        skill[skill<0] = 0
        skill[skill>1] = 1
    elif method == 'dispersion':
        #skill = 1.0 - np.cumsum(rel_dispers)/np.cumsum(dispers)
        skill = 1.0 - (2*rel_dispers)/(dispers)
        # check limits and return skill
        skill[skill<0] = 0
        skill[skill>1] = 1
    elif method == 'sep':
        skill = sep
    else:
        print('Don''t recognize {} method.'.format(method))
        import sys
        sys.exit()
    return skill

def calculateSkillScores(dataObs, dataMod) :
	sep = np.zeros([len(dataObs.Latitude)-1,])
	disp = np.zeros_like(sep)
	disp_mod = np.zeros_like(sep)
	dist = np.zeros_like(sep)
	distm = np.zeros_like(sep)

	pos_obs = np.column_stack([dataObs.Latitude, dataObs.Longitude])
	pos_mod = np.column_stack([dataMod.Latitude, dataMod.Longitude])

	for i in range(len(sep)):
		sep[i] = vincenty(pos_obs[i+1,:], pos_mod[i+1,:]).meters
		disp[i] = vincenty(pos_obs[0,:], pos_obs[i+1,:]).meters
		disp_mod[i] = vincenty(pos_mod[0,:], pos_mod[i+1,:]).meters
		dist[i] = vincenty(pos_obs[i,:], pos_obs[i+1,:]).meters
		distm[i] = vincenty(pos_mod[i,:], pos_mod[i+1,:]).meters

	## calculate skill scores
	# first liu (Liu and Weissenberg 2011)
	liu = 1.0 - np.cumsum(sep) / np.cumsum(np.cumsum(dist))
	liu[liu < 0.0] = 0.0
	# next molcard
	molcard = 1.0 - sep/disp
	molcard[molcard < 0.0] = 1.0
	# Toner et al. (2002)
	toner = 1.0 - sep / np.cumsum(dist)
	toner[toner < 0.0] = 0.0
	# a new one based on area
	area_skill = 1.0 - np.cumsum(sep*dist) / np.cumsum(disp*dist)
	area_skill[area_skill < 0.0] = 0.0
	# area ratio
	ss = np.cumsum(sep) / np.cumsum(np.cumsum(dist))
	ssa = np.cumsum(sep*dist) / np.cumsum(disp*dist)
	ratio = ssa/ss

	return liu, molcard, toner, area_skill, ratio, sep*1e-3, disp_mod*1e-3, disp*1e-3, np.cumsum(dist)*1e-3, np.cumsum(distm)*1e-3

def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
	 	  theta = atan2(sin(dlong)*cos(lat2),
		            cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlong)
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    #if (type(pointA) != tuple) or (type(pointB) != tuple):
    #    raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180 to + 180 which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing
