from model import Energy
import matplotlib.pyplot as plt

arcticdem_path = "/home/tepex/AARI/Glaciology_2019/lighting/source/dem.tif"
glacier_outlines_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/shp/aldegonda_outlines_2018/aldegonda_outlines_2018.shp"
albedo_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/gridded/L8_albedo_20190727_crop.tif"  # constant albedo for the whole year
potential_insolation_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/gridded/total_insolation_July.tif"

aws_file = "/home/tepex/PycharmProjects/energy/aws/test_aws_data.csv"
albedo_maps = {
	"20190727": "/home/tepex/AARI/Glaciers/Aldegonda albedo/gridded/L8_albedo_20190727_crop.tif",
	"20190803": "/home/tepex/AARI/Glaciers/Aldegonda albedo/gridded/L8_albedo_20190803_crop.tif"
}

"""
Let's test the model
"""
t_air = 2.46  # degrees Celsius
wind_speed = 1.81  # meters per second
rel_humidity = 0.858  # 0-1
air_pressure = 990.14  # hPa
cloudiness = 0.8  # 0-1
incoming_shortwave = 250  # W per sq meter (J per second per sq meter)
z = 1.6  # AWS sensors height above ice surface, meters
elev_aws = 290
xy_aws = (478342, 8655635)  # EPSG:32633
#################
e = Energy(arcticdem_path, glacier_outlines_path, albedo_path, potential_insolation_path)

e.model(aws_file=aws_file, albedo_maps=albedo_maps, z=1.6, elev_aws=elev_aws, xy_aws=xy_aws)
