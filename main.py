from model import Energy
import os

arcticdem_path = "/home/tepex/AARI/Glaciology_2019/lighting/source/dem.tif"
glacier_outlines_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/shp/aldegonda_outlines_2018/aldegonda_outlines_2018.shp"
albedo_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/gridded/L8_albedo_20190727_crop.tif"  # constant albedo for the whole year
insolation_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/gridded/total_insolation_July.tif"

e = Energy(arcticdem_path, glacier_outlines_path, albedo_path, insolation_path)
e.show_me(e.base_dem_array)
# e.show_me(e.constant_albedo)
# e.show_me(e.constant_insolation)
"""
Let's test the model
"""
# t_air, wind_speed, rel_humidity, air_pressure, cloudiness, incoming_shortwave, albedo
t_air = 3
wind_speed = 2
rel_humidity = 0.2
air_pressure = 1013
cloudiness = 0.7
incoming_shortwave = 300
albedo = 0.30
z = 1.6
#################
influx = e.calc_heat_influx(t_air, wind_speed, rel_humidity, air_pressure, cloudiness, incoming_shortwave, albedo, z)
melt = e.calc_ice_melt(influx, 916.7, 30)
print(("Test melt equals %.3f m w.e.") % melt)
