import numpy as np
import pandas as pd
import scipy as sp

def running_mean(x, n):
    # calculating running mean of x for every n segments
    xf = np.convolve(x, np.ones((n,))/n, mode='same')
    xf[0] = np.mean(x[:2])
    xf[-1] = np.mean(x[-2:])
    return xf

def find_peaks(data, smooth_radius, threshold=1e-5):
    data = sp.ndimage.uniform_filter(data, smooth_radius)
    thresh = data > threshold
    filled = sp.ndimage.morphology.binary_fill_holes(thresh)
    coded_peaks, num_peaks = sp.ndimage.label(filled)
    data_slices = sp.ndimag.find_objects(coded_peaks)
    return data_slices

# a function to return an ordered list of keys for which in the right time interval, ordered by time
def sort_dict_keys_by_date(dict_in):
    list_key_datetime = []

    for crrt_key in dict_in:
        crrt_datetime = dict_in[crrt_key]['datetime']
        list_key_datetime.append((crrt_key, crrt_datetime))

    sorted_keys = sorted(list_key_datetime, key=lambda x: x[1])
    sorted_keys = [x for (x, y) in sorted_keys]

    return(sorted_keys)


def bootstrap(data, n=1000, func=np.mean):
   """
   Generate `n` bootstrap samplse, evaluating `func`
   at each resampling. `bootstrap` returns a function,
   which can be called to obtain confidence intervals
   of interest.
   """
   simulations = list()
   sample_size = len(data)
   xbar_init = np.mean(data)
   for c in range(n):
      itersample = np.random.choice(data, size=sample_size, replace=True)
      simulations.append(func(itersample))
   simulations.sort()
   def ci(p):
      """
      Return 2-sided symmetric confidence interval specified
      by p.
      """
      u_pval = (1+p)/2.0
      l_pval = (1-u_pval)
      l_indx = int(np.floor(n*l_pval))
      u_indx = int(np.floor(n*u_pval))
      return(simulations[l_indx], simulations[u_indx])
   return(ci)

# a function to return only the keys that correspond to a given kind of data (status / spectrum)

def meridional_curvature(phi):
	''' calculate meridional curvature for a given latitude (phi)'''
	a = 6378.137 # semi-major axis
	b = 6356.7523 # semi-minor axis

	return (a*b)**2 / ( (a*np.cos(phi))**2 + (b*np.sin(phi))**2 )**1.5

def prime_curvature(phi):
	''' calculate prime curvature for a given latitude (phi)'''
	a = 6378.137 # semi-major axis
	b = 6356.7523 # semi-minor axis
	return a**2 / ( (a*np.cos(phi))**2 + (b*np.sin(phi))**2 )**0.5

def calculate_separation_distance(dataObs1, dataObs2):
   ''' calculate separation distance between two data frames'''
   lat1 = dataObs1.Latitude
   lon1 = dataObs1.Longitude
   lat2 = dataObs2.Latitude
   lon2 = dataObs2.Longitude
   phi = np.mean(np.deg2rad(lat1))
   dlat = np.deg2rad(lat1-lat2)
   dlon = np.deg2rad(lon1-lon2)
   M = meridional_curvature(phi)
   N = prime_curvature(phi) * np.cos(phi)
   dx = N*dlon
   dy = M*dlat
   dist = np.sqrt(dx**2 + dy**2)
   return dist

def calculate_distance(lon2, lat2, lo, la):
    dlon = np.deg2rad(lon2 - lo)
    dlat = np.deg2rad(lat2 - la)
    phi = np.deg2rad(la)
    M = meridional_curvature(phi)
    N = prime_curvature(phi)*np.cos(phi)
    dx, dy = N*dlon, M*dlat
    return np.sqrt(dx**2 + dy**2)

def lonlat2xykm(lon2, lat2, lo, la):
    dlon = np.deg2rad(lon2-lo)
    dlat = np.deg2rad(lat2-la)
    phi = np.deg2rad(la)
    M = meridional_curvature(phi)
    N = prime_curvature(phi)*np.cos(phi)
    dx, dy = N*dlon, M*dlat
    return dx, dy

def drogue_depth(ID):
   '''dictionary with a list of drogue depths'''
   if ID.startswith('Osker') or ID.startswith('ISPHERE'):
      drogue_depth = -0.0
   elif ID.startswith('Roby'):
      drogue_depth = -0.1
   elif ID.startswith('SCT'):
      drogue_depth = -0.2
   elif ID.startswith('CODE'):
      drogue_depth = -0.6
   elif ID.startswith('SVP'):
      drogue_depth = -15.0
   else:
      print('Incorrect ID of {}'.format(ID))
      drogue_depth = np.NaN
   return drogue_depth

def calculate_drifter_speed_forward(dataObs):
    npts = len(dataObs)
    vel = np.zeros([npts,2])
    pos = np.column_stack([dataObs.Latitude, dataObs.Longitude])
    for i in range(npts):
        if i == npts-1:
            dt = (dataObs.index[i]-dataObs.index[i-1]).total_seconds() / 3600.0
            phi = np.deg2rad(0.5*(pos[i-1,0]+pos[i,0]))
            dlat = np.deg2rad(pos[i,0] - pos[i-1,0])
            dlon = np.deg2rad(pos[i,1] - pos[i-1,1])
            M = meridional_curvature(phi)
            N = prime_curvature(phi) * np.cos(phi)
            vel[i,0] = N * dlon / dt
            vel[i,1] = M * dlat / dt
        else:
            dt = (dataObs.index[i+1]-dataObs.index[i]).total_seconds() / 3600.0
            phi = np.deg2rad(0.5*(pos[i,0]+pos[i+1,0]))
            dlat = np.deg2rad(pos[i+1,0] - pos[i,0])
            dlon = np.deg2rad(pos[i+1,1] - pos[i,1])
            M = meridional_curvature(phi)
            N = prime_curvature(phi) * np.cos(phi)
            vel[i,0] = N * dlon / dt
            vel[i,1] = M * dlat / dt
    return vel

def calculate_drifter_speed_old(dataObs):
   ''' calculate eastward and northward velocities in m/s from drifter locations'''
   npts = len(dataObs)
   vel = np.zeros([npts,2])
   pos = np.column_stack([dataObs.Latitude, dataObs.Longitude])
   for i in range(npts):
      if i == 0:
         dt = (dataObs.index[i+1]-dataObs.index[i]).total_seconds() / 3600.0
         phi = np.deg2rad(0.5*(pos[i,0]+pos[i+1,0])) # mean latitude
         dlat = np.deg2rad(pos[i+1,0] - pos[i,0])
         dlon = np.deg2rad(pos[i+1,1] - pos[i,1])
         M = meridional_curvature(phi)
         N = prime_curvature(phi) * np.cos(phi)
         vel[i,0] = N * dlon / dt
         vel[i,1] = M * dlat / dt
      elif i == npts-1:
         dt = (dataObs.index[i]-dataObs.index[i-1]).total_seconds() / 3600.0
         phi = np.deg2rad(0.5*(pos[i-1,0]+pos[i,0]))
         dlat = np.deg2rad(pos[i,0] - pos[i-1,0])
         dlon = np.deg2rad(pos[i,1] - pos[i-1,1])
         M = meridional_curvature(phi)
         N = prime_curvature(phi) * np.cos(phi)
         vel[i,0] = N * dlon / dt
         vel[i,1] = M * dlat / dt
      else:
         phi = np.deg2rad(pos[i,0])
         M = meridional_curvature(phi)
         N = prime_curvature(phi) * np.cos(phi)
         dt = (dataObs.index[i+1]-dataObs.index[i-1]).total_seconds() / 3600.0
         dlat = np.deg2rad(pos[i+1,0] - pos[i-1,0])
         dlon = np.deg2rad(pos[i+1,1] - pos[i-1,1])
         vel[i,0] = N * dlon/dt
         vel[i,1] = M * dlat/dt

   # convert vel from km/h -> m/s
   vel /= 3.6
   return vel

def calculate_drifter_speed(dataObs):
   ''' calculate eastward and northward velocities in m/s from drifter locations'''
   npts = len(dataObs)
   vel = np.zeros([npts,2])
   pos = np.column_stack([dataObs.Latitude, dataObs.Longitude])
   for i in range(npts):
      if i < npts-1:
         dt = (dataObs.index[i+1]-dataObs.index[i]).total_seconds() / 3600.0
         phi = np.deg2rad(0.5*(pos[i,0]+pos[i+1,0])) # mean latitude
         dlat = np.deg2rad(pos[i+1,0] - pos[i,0])
         dlon = np.deg2rad(pos[i+1,1] - pos[i,1])
         M = meridional_curvature(phi)
         N = prime_curvature(phi) * np.cos(phi)
         vel[i,0] = N * dlon / dt
         vel[i,1] = M * dlat / dt
      else:
         dt = (dataObs.index[i]-dataObs.index[i-1]).total_seconds() / 3600.0
         phi = np.deg2rad(0.5*(pos[i-1,0]+pos[i,0]))
         dlat = np.deg2rad(pos[i,0] - pos[i-1,0])
         dlon = np.deg2rad(pos[i,1] - pos[i-1,1])
         M = meridional_curvature(phi)
         N = prime_curvature(phi) * np.cos(phi)
         vel[i,0] = N * dlon / dt
         vel[i,1] = M * dlat / dt

   # convert vel from km/h -> m/s
   vel /= 3.6
   return vel

def grid_angle(lons, lats):
   """ Find the angle between longitude grid lines and East.
       theta = pi/2 - (atan2(sin(dlong)*cos(lat2), cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlong)))
       ueast = ux*cos(theta) - vy*sin(theta)
       vnorth = ux*sin(theta) + vy*cos(theta)
       input is lons and lats with shape (y,x)
       if shape is (x,y) then see grid_angle_rev
   """
   dlons = np.radians(lons[:,1:] - lons[:, 0:-1])
   lat1 = np.radians(lats[:, 0:-1])
   lat2 = np.radians(lats[:, 1:])
   theta = np.empty_like(lons)
   x = np.sin(dlons)*np.cos(lat2)
   y = np.cos(lat1) * np.sin(lat2) - \
      (np.sin(lat1)* np.cos(lat2) * np.cos(dlons))
   # Extend theta by copying first column
   theta[:, 1:] = np.arctan2(x,y)
   theta[:, 0] = theta[:,1]
   # Theta is the angle with North so subtract from pi/2 for angle with East
   return np.pi/2 -theta

def grid_angle_rev(lons, lats):
   '''calculate grid angle but lons(x, y) and lats(x,y) instead of lons(y,x)'''

   dlons = np.radians(lons[1:,:] - lons[0:-1,:])
   lat1 = np.radians(lats[0:-1, :])
   lat2 = np.radians(lats[1:, :])
   theta = np.empty_like(lons)
   x = np.sin(dlons)*np.cos(lat2)
   y = np.cos(lat1) * np.sin(lat2) - \
      (np.sin(lat1)* np.cos(lat2) * np.cos(dlons))
   # Extend theta by copying first column
   theta[1:, :] = np.arctan2(x,y)
   theta[0, :] = theta[1, :]
   # Theta is the angle with North so subtract from pi/2 for angle with East
   return np.pi/2 - theta

def min_ellipse(P, tolerance=1e-4):
    '''A function to calculate the minimuym area ellipse which encloses all the points'''
    # should be 2,N coordinates for N data points
    d, N = P.shape
    if d != 2:
        print('!! Wrong dmiensions !! Needs to be 2xN.')

    Q = np.ones((d+1, N))
    Q[0:d, :] = P
    u = (1.0/N) * np.ones((N,))

    count = 1
    err = 1

    while err > tolerance:
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, np.linalg.inv(X)), Q))
        maximum = np.max(M)
        j = np.argmax(M)

        step_size = (maximum - d - 1.0) / ((d+1)*(maximum-1.0))

        new_u = (1.0 - step_size) * u
        new_u[j] += step_size

        err = np.linalg.norm(new_u - u)

        count += 1
        u = new_u

    # define centre
    C = np.dot(P, u)
    # and A-matrix
    U = np.diag(u)
    pup = np.dot(np.dot(P,U), P.T)
    pupu = np.dot((np.dot(P,u)), (np.dot(P,u)).T)
    A = (1.0/d) * np.linalg.inv(pup - pupu)

    return C, A

def std_ellipse(P):
    # function to calculate ellipse based on standard deviation
    d, N = P.shape
    if d != 2:
        print('!! Wrong dimensions !! Needs to be 2xN')

    Pm = P.mean(axis=1)
    Pdev = (P.T - Pm).T
    a1 = np.sum(Pdev[0,:]**2) - np.sum(Pdev[1,:]**2)
    a2 = np.sqrt( (np.sum(Pdev[0,:]**2)-np.sum(Pdev[1,:]**2))**2 + \
            4.0*(np.sum(Pdev[0,:]*Pdev[1,:])**2) )
    a3 = 2*np.sum(Pdev[0,:]*Pdev[1,:])
    th1 = np.arctan( (a1+a2)/a3 )
    th2 = np.arctan( (a1-a2)/a3 )
    std_theta = th2
    sigx = np.sqrt( np.mean( (Pdev[1,:]*np.sin(std_theta)+Pdev[0,:]*np.cos(std_theta))**2 ) )
    sigy = np.sqrt( np.mean( (Pdev[1,:]*np.cos(std_theta)-Pdev[0,:]*np.sin(std_theta))**2 ) )

    return sigx, sigy, std_theta
    
    
    
