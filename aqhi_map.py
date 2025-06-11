import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import requests
from io import StringIO

# 1. Load CSV data from GitHub (make sure it's raw URL)
url = 'https://raw.githubusercontent.com/your-username/your-repo/main/data.csv'
response = requests.get(url)
df = pd.read_csv(StringIO(response.text))

# 2. Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='EPSG:4326')
gdf = gdf.to_crs(epsg=3857)  # convert to metric

# 3. Generate grid
xmin, ymin, xmax, ymax = gdf.total_bounds
cellsize = 1000  # in meters
x = np.arange(xmin, xmax, cellsize)
y = np.arange(ymin, ymax, cellsize)
xx, yy = np.meshgrid(x, y)
grid_points = np.c_[xx.ravel(), yy.ravel()]

# 4. Perform IDW
def idw(xy, values, grid_xy, power=2):
    dist = np.sqrt(((grid_xy[:, None, :] - xy[None, :, :])**2).sum(axis=2))
    with np.errstate(divide='ignore'):
        weights = 1 / dist**power
    weights[dist == 0] = 1e10  # handle divide by zero
    weights[dist == 0] = weights.max()
    weights_sum = weights.sum(axis=1)
    interp_values = (weights @ values) / weights_sum
    return interp_values

xy = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
values = gdf['value_column'].values  # replace with actual column
grid_values = idw(xy, values, grid_points)

# 5. Convert grid points to polygons
polygons = []
vals = []
for i in range(len(grid_points)):
    x0, y0 = grid_points[i]
    poly = Polygon([
        (x0, y0),
        (x0 + cellsize, y0),
        (x0 + cellsize, y0 + cellsize),
        (x0, y0 + cellsize)
    ])
    polygons.append(poly)
    vals.append(grid_values[i])

grid_gdf = gpd.GeoDataFrame({'value': vals}, geometry=polygons, crs=gdf.crs)

# 6. Save or visualize
grid_gdf.to_file("interpolated_grid.geojson", driver='GeoJSON')
