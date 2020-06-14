from model import Energy
import matplotlib.pyplot as plt

arcticdem_path = "/home/tepex/AARI/Glaciology_2019/lighting/source/dem.tif"
glacier_outlines_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/shp/aldegonda_outlines_2018/aldegonda_outlines_2018.shp"
albedo_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/gridded/L8_albedo_20190727_crop.tif"  # constant albedo for the whole year
insolation_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/gridded/total_insolation_July.tif"

e = Energy(arcticdem_path, glacier_outlines_path, albedo_path, insolation_path)
# e.show_me(e.base_dem_array)
# e.show_me(e.constant_albedo)
# e.show_me(e.constant_insolation)
"""
Let's test the model
"""
# t_air, wind_speed, rel_humidity, air_pressure, cloudiness, incoming_shortwave, albedo, z
t_air = 3  # degree Celsius
wind_speed = 2.5  # meters per second
rel_humidity = 0.85  # 0-1
air_pressure = 1013  # hPa
cloudiness = 0.7  # 0-1
incoming_shortwave = 300  # W per sq meter (J per second per sq meter)
albedo = 0.30  # 0-1
z = 1.6  # AWS sensors height above ice surface, meters
aws_coords = (14, 78, 190)  # coordinates of weather station
#################
influx = e.calc_heat_influx(t_air, wind_speed, rel_humidity, air_pressure, cloudiness, incoming_shortwave, albedo, z, aws_coords[2])
melt = e.calc_ice_melt(influx, 916.7, 30)
plt.imshow(influx)
plt.colorbar()
plt.title("Test heat influx, W per sq m")
plt.show()

plt.imshow(melt)
plt.colorbar()
plt.title("Test melt, m w.e.")
plt.show()

"""
air_t_array = e.interpolate_air_t(t_air, aws_coords[2])
plt.imshow(air_t_array)
plt.colorbar()
plt.title("Air temperature grid")
plt.show()
"""
