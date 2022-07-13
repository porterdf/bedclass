# import QgsProject
# import qgis
# import processing

#root = QgsProject.instance().layerTreeRoot()
#for layer in root.children():
#  print(layer.name())

root = QgsProject.instance().layerTreeRoot()

### RUNTIME PARAMETERS
#potentialfield = 'MAG'
#input_layer = 'Users/dporter/Documents/Research/Projects/bedclass/data/mag_upward_200_CutByGL.geojson'
#potentialfield = 'GRAV'
#input_layer = 'Users/dporter/Documents/Research/Projects/bedclass/data/grav_upward_30_CutByGL.geojson'
#potentialfield = 'GRAVMAG'
#input_layer = '/Users/dporter/Documents/Research/Projects/bedclass/data/gravmag_upward+GL+catch_pruned.geojson'
potentialfield = 'MAG_dziadek'
input_layer = 'Users/dporter/Documents/Research/Projects/bedclass/data/ASE_Mag_Dziadek_150int_vector.geojson'
#
#
#input_layer = 'Zonal Statistics'
result_layer = input_layer
unique_field = 'fid'

################
## Extract only "Grounded" ice
# Algorithm 'Extract by attribute' starting…
params = { 'FIELD' : 'groundediceNAME', 
    'INPUT' : input_layer, 
    'OPERATOR' : 0, 
    'OUTPUT' : 'TEMPORARY_OUTPUT', 
    'VALUE' : 'Grounded',
    }
result = processing.run("native:extractbyattribute", params)
result_layer = result['OUTPUT']

################
## Add geometry
#params = {'INPUT':result_layer,
#    'CALC_METHOD':0,
#    'OUTPUT':'TEMPORARY_OUTPUT',
#    }
#result = processing.run("qgis:exportaddgeometrycolumns", params)
#result_layer = result['OUTPUT']

################
## Kirsty's zones
## gravity 
prefix = 'grav_zones'
params = {'INPUT_RASTER': '/Users/dporter/Documents/Research/Projects/bedclass/data/grav_upward_class_30_amund_meter.tif',
    'INPUT': result_layer,
    'OUTPUT': 'memory:',
    'RASTER_BAND' : 1,
    'COLUMN_PREFIX': prefix+'_', 
    'STATISTICS' : [2,3],
    }
result = processing.run("native:zonalstatisticsfb", params)
result_layer = result['OUTPUT']

################
## Kirsty's zones
## magnetics 
prefix = 'mag_zones'
params = {'INPUT_RASTER': '/Users/dporter/Documents/Research/Projects/bedclass/data/mag_upward_class_200_amund.tif',
    'INPUT': result_layer,
    'OUTPUT': 'memory:',
    'RASTER_BAND' : 1,
    'COLUMN_PREFIX': prefix+'_', 
    'STATISTICS' : [2,3],
    }
result = processing.run("native:zonalstatisticsfb", params)
result_layer = result['OUTPUT']

################
## AntGG 10 km
prefix = 'bouger' # layer.name()[:4]
params = {'INPUT_RASTER' : '/Users/dporter/data/Antarctic/Quantarctica3/Geophysics/ANTGG/ANTGG_BouguerAnomaly_10km.tif', 
    'INPUT': result_layer,
    'OUTPUT': 'memory:',
    'RASTER_BAND' : 1,
    'COLUMN_PREFIX': prefix+'_', 
    'STATISTICS' : [3,4],
    }
result = processing.run("native:zonalstatisticsfb", params)
result_layer = result['OUTPUT']

################
## MAG (Dziadek)
prefix = 'Dziadek' # layer.name()[:4]
params = {'INPUT_RASTER' : '/Users/dporter/data/Antarctic/Dziadek-etal_2021/ASE_MagneticCompilation_Dziadeketal_250m.tif', 
    'INPUT': result_layer,
    'OUTPUT': 'memory:',
    'RASTER_BAND' : 1,
    'COLUMN_PREFIX': prefix+'_', 
    'STATISTICS' : [3,4],
    }
result = processing.run("native:zonalstatisticsfb", params)
result_layer = result['OUTPUT']

################
## ICESat II
prefix = 'dmdt' # layer.name()[:4]
params = {'INPUT_RASTER' : '/Users/dporter/data/Antarctic/Satellite/icesat/ICESat1_ICESat2_mass_change_updated_2_2021/dmdt/ais_dmdt_grounded.tif', 
    'INPUT': result_layer,
    'OUTPUT': 'memory:',
    'RASTER_BAND' : 1,
    'COLUMN_PREFIX': prefix+'_', 
    'STATISTICS' : [3,4],
    }
result = processing.run("native:zonalstatisticsfb", params)
result_layer = result['OUTPUT']

################
## PISM Beta
prefix = 'beta' # layer.name()[:4]
params = {'INPUT_RASTER' : '/Users/dporter/Documents/Research/Projects/bedclass/data/PISM_beta_log10.tif', 
    'INPUT': result_layer,
    'OUTPUT': 'memory:',
    'RASTER_BAND' : 1,
    'COLUMN_PREFIX': prefix+'_', 
    'STATISTICS' : [3,4],
    }
result = processing.run("native:zonalstatisticsfb", params)
result_layer = result['OUTPUT']


## Save to a file
outfile = f'/Users/dporter/Documents/Research/Projects/bedclass/data/zonalstats_{potentialfield}.json'
qgis.core.QgsVectorFileWriter.writeAsVectorFormat(result_layer,
    outfile, 
    'utf-8', 
    result_layer.crs(), 
    'GeoJson')

## Add to Map
#QgsProject.instance().addMapLayer(new_layer) # result_layer
newlayer = qgis.core.QgsVectorLayer(outfile, f'Zonal Stats {potentialfield}', "ogr")
print(newlayer.isValid())
qgis.core.QgsProject.instance().addMapLayer(newlayer)