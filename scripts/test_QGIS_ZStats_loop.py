#root = QgsProject.instance().layerTreeRoot()
#for layer in root.children():
#  print(layer.name())

root = QgsProject.instance().layerTreeRoot()

input_layer = 'Zonal Statistics'
result_layer = input_layer
unique_field = 'fid'

# Iterate through all raster layers
for layer in root.children():
  if layer.name().startswith('grav_upward_'):
    # Run Zonal Stats algorithm

    prefix = 'bouger' # layer.name()[:4]
    params = {'INPUT_RASTER': layer.name(),
        'RASTER_BAND': 1, 'INPUT': input_layer,
        'COLUMN_PREFIX': prefix+'_', 'STATISTICS': [2],
        'OUTPUT': 'memory:'
        }
    result = processing.run("native:zonalstatisticsfb", params)

    result_layer = result['OUTPUT']
    # Breaking out of loop to demonstrate the zonalstatistics algorithm.
    break

QgsProject.instance().addMapLayer(result_layer)