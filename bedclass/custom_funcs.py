def read_DICE_matfile(infile):
    from scipy.io import loadmat
    import pandas as pd

    ## Constants
    cAir = 299792458
    cIce = 1.68e8
    
    ## Load mat file
    matdata = loadmat(infile)
    
    # dict_keys(['__header__', '__version__', '__globals__', 'pickername', 'NOTE_VertScale', 'Time', 'Surf_Elev', 'GPS_time', 'FlightElev', 'SurfTime', 'Lat', 'Lon', 'Pixel', 'PickTime', 'X', 'Y', 'Distance', 'xdisp', 'Depth', 'Bright', 'MultipleBright', 'NoiseFloor', 'Notes', 'Data', 'VertScale'])
    
    Data = matdata['Data'][:].squeeze()
    X = matdata['X'][:].squeeze()*1e3
    Y = matdata['Y'][:].squeeze()*1e3
    lat = matdata['Lat'][:].squeeze()
    lon = matdata['Lon'][:].squeeze()

    Elev = matdata['FlightElev'][:].squeeze()
    Surf = matdata['SurfTime'][:].squeeze()  # SurfTime, or Surf_Elev
    PickTime = matdata['PickTime'][:].squeeze()
    Surf_elev = matdata['Surf_Elev'][:].squeeze()
    Flightelev = matdata['FlightElev'][:].squeeze()  
    Time = matdata['Time'][:].squeeze()
    Depth = matdata['Depth'][:].squeeze()
    
    GPS_time = matdata['GPS_time'][:].squeeze()
    Pixel = matdata['Pixel'][:].squeeze()  
    Distance = matdata['Distance'][:].squeeze()
    Bright = matdata['Bright'][:].squeeze()
    
    # Derive
    PickDepth = -(.5*cIce)*PickTime
    SurfDepth = (.5*cAir)*Surf
    
    # Write as dataframe
    df = pd.DataFrame(data=[GPS_time, Time, 
                            PickTime, Surf_elev, 
                            X, Y, 
                            lat, lon,
                            Elev, Surf, 
                            Flightelev, Depth,
                            Bright, Distance,
                            PickDepth, SurfDepth]).T
    
    df.rename(columns={0: 'GPS_time', 1: 'Time',
                       2: 'PickTime', 3: 'Surf_elev', 
                       4: 'EPSG_X', 5: 'EPSG_Y',
                       6: 'lat', 7: 'lon',
                       8: 'Elev', 9: 'Surf',
                       10: 'FlightElev', 11: 'Depth',
                       12: 'Bright', 13: 'Distance',
                       14: 'PickDepth', 15: 'SurfDepth'},
                inplace=True)
    
    return df


def print_raster(raster):
    print(
        f"shape: {raster.shape}\n"
#         f"resolution: {raster.resolution()}\n"
        f"bounds: {raster.bounds}\n"
#         f"sum: {raster.sum().item()}\n"
        f"CRS: {raster.crs}\n"
    )
    