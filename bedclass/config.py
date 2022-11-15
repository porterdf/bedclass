# config.py

from pathlib import Path  # pathlib is seriously awesome!

## Constants
# m/s
cAir = 299792458
cIce = 1.68e8

## Data paths
base_dir = Path('/Users/dporter/data')
data_dir = Path('Antarctic/ROSETTA/radar')
data_path = base_dir / data_dir
# data_path = base_dir / data_dir / 'my_file.csv'  # use feather files if possible!!!


