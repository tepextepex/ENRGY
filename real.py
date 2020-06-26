from model import Energy
import matplotlib.pyplot as plt
import time

arcticdem_path = "/home/tepex/AARI/Glaciology_2019/lighting/source/dem.tif"
glacier_outlines_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/shp/aldegonda_outlines_2018/aldegonda_outlines_2018.shp"
potential_insolation_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/gridded/total_insolation_August.tif"  # kW*h per month
out_file = "/home/tepex/PycharmProjects/energy/aws/out.csv"

aws_file = "/home/tepex/PycharmProjects/energy/aws/DATA_08_2019.csv"

albedo_maps = {
	"20190727": "/home/tepex/AARI/Glaciers/Aldegonda albedo/albedo_without_shadows/albedo_20190727.tif",
	"20190803": "/home/tepex/AARI/Glaciers/Aldegonda albedo/albedo_without_shadows/albedo_20190803.tif",
	"20190821": "/home/tepex/AARI/Glaciers/Aldegonda albedo/albedo_without_shadows/albedo_20190821.tif",
	"20190831": "/home/tepex/AARI/Glaciers/Aldegonda albedo/albedo_without_shadows/albedo_20190821.tif"  # NOT a mistake. Using the same file gives you constant albedo over the time
}

"""
Let's test the model
"""
z = 1.6  # AWS sensors height above ice surface, [m]
elev_aws = 290  # AWS elevation (absolute, NOT above the ice surface), [m]
xy_aws = (478342, 8655635)  # x and y coordinates of AWS in EPSG:32633, [m]

e = Energy(arcticdem_path, glacier_outlines_path, potential_insolation_path)

start_time = time.time()
e.model(aws_file=aws_file, out_file=out_file, albedo_maps=albedo_maps, z=1.6, elev_aws=elev_aws, xy_aws=xy_aws)

exec_time = time.time() - start_time
m = exec_time // 60
s = exec_time % 60
print("Done in %d m %.1f s" % (m, s))
