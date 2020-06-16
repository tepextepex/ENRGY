from model import Energy
import matplotlib.pyplot as plt

arcticdem_path = "/home/tepex/AARI/Glaciology_2019/lighting/source/dem.tif"
glacier_outlines_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/shp/aldegonda_outlines_2018/aldegonda_outlines_2018.shp"
albedo_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/gridded/L8_albedo_20190727_crop.tif"  # constant albedo for the whole year
potential_insolation_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/gridded/total_insolation_July.tif"

"""
Let's test the model
"""
t_air = 3  # degrees Celsius
wind_speed = 2.5  # meters per second
rel_humidity = 0.85  # 0-1
air_pressure = 1000  # hPa
cloudiness = 0.8  # 0-1
incoming_shortwave = 250  # W per sq meter (J per second per sq meter)
z = 1.6  # AWS sensors height above ice surface, meters
aws_coords = (14, 78, 190)  # coordinates of weather station
days = 30
#################
e = Energy(arcticdem_path, glacier_outlines_path, albedo_path, potential_insolation_path)
e.set_aws_data(t_air, wind_speed, rel_humidity, air_pressure, cloudiness, incoming_shortwave, z, aws_coords[2])
e.run(days)
