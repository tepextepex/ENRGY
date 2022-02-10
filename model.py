import os
import csv
import numpy as np
from datetime import datetime

from parameter_classes import CONST
from parameter_classes import AwsParams, DistributedParams, OutputRow
from turbo import calc_turbulent_fluxes
from raster_utils import show_me, load_raster, export_array_as_geotiff, OUT_DIR, get_value_by_real_coords
from interpolator import interpolate_array
from saga_lighting import simulate_lighting, cleanup_sgrd
from msm import calc_melt, tick


class Energy:
    def __init__(self, base_dem_path, glacier_outlines_path):

        self._init_constants()

        self.current_date_str = None
        self.input_list = []
        self.output_row = None

        self.png_export = 1  # will export every n-th png, default is exporting on the each time step
        self.result_export_dates = None

        # for potential incoming solar radiation:
        self.export_potential = False
        self.use_precomputed = False

        self.aws = None
        self.params = None
        self.albedo = None
        self.albedo_arrays = None

        self.use_msm = False
        self.layer_temperatures = None
        self.layer_depths = []

        self.potential_incoming_sr_path = None
        self.potential_incoming_radiation = None
        self.incoming_shortwave = None

        self.base_dem_path = base_dem_path
        self.outlines_path = glacier_outlines_path

        print("Loading base DEM...")
        self.base_dem_array, self.geotransform, self.projection = load_raster(base_dem_path, self.outlines_path,
                                                                              v=False)
        self.total_snow_melt_array = np.zeros_like(self.base_dem_array, dtype=np.float32)
        self.total_ice_melt_array = np.zeros_like(self.base_dem_array, dtype=np.float32)

        self.swe_array = np.zeros_like(self.base_dem_array,
                                       dtype=np.float32)  # snow cover may be added after via add_snow method
        # DEBUG:
        # self.swe_array.fill(0.005)

    def _init_constants(self):
        self.CONST = CONST

    def add_snow(self, swe_map_path):
        print("Initialized snow cover state (SWE) from %s" % swe_map_path)
        self.swe_array = load_raster(swe_map_path, self.outlines_path, v=False)[0]

    def add_msm(self, depths, temperatures, elev_aws):
        print("Initializing subsurface model...")

        self.use_msm = True
        self.layer_depths = depths
        self.layer_temperatures = []

        def extrapolate(value, v_gradient=None):
            delta_dem = self.base_dem_array - elev_aws
            if v_gradient is None:
                v_gradient = -0.008  # 8 degrees Celsius or Kelvins per 1 km, year 2021 value
            return value + delta_dem * v_gradient

        for t_point in temperatures:
            t_distributed = extrapolate(t_point)
            t_distributed[t_distributed > 0] = 0.0  # ice temperature is limited by melting point
            self.layer_temperatures.append(t_distributed)
        # print(self.layer_depths)

        i = 0
        for temp_array in self.layer_temperatures:
            show_me(temp_array, title="Layer %s temperature" % i, dir="Glacier body temperature")
            i += 1

    def add_result_export_dates(self, date_str_list):
        dts = [s + " 12:00:00" for s in date_str_list]
        self.result_export_dates = dts

    def model(self, aws_file=None, out_file=None, albedo_maps=None, z=2.0, elev_aws=0.0, xy_aws=None, solar_only=False,
              png=True, v=True):
        if (aws_file is not None) and (albedo_maps is not None):
            # loading albedo maps from geotiff files into arrays:
            self.albedo_arrays = {}
            for key in albedo_maps:  # albedo_maps contains file paths
                self.albedo_arrays[key] = load_raster(albedo_maps[key], self.outlines_path, remove_outliers=True, v=v)[
                    0]

            self.fill_header(out_file)

            self.input_list = self.read_input_file(aws_file)
            for i in range(len(self.input_list)):
                png = True if i % self.png_export == 0 else False

                row = self.input_list[i]
                print("\nProcessing %s..." % row["DATE"])
                self.current_date_str = row["DATE"]

                try:
                    time_step = self.get_time_step(self.input_list, i, "%Y%m%d")
                except ValueError:
                    time_step = self.get_time_step(self.input_list, i, "%Y%m%d %H:%M:%S")
                # print("########## TIME STEP IS: %s" % time_step)  # DEBUG

                # setting meteo parameters for the current date:
                r_hum = self.heuristic_unit_guesser(float(row["HUMID"]), 100)
                # cld = self.heuristic_unit_guesser(float(row["CLOUDINESS"]), 10)  # causes a bug when value = 1: should be treated as 0.1, but returned as 1.0 instead
                cld = float(row["CLOUDINESS"])

                if self.layer_temperatures is None:
                    t_surf = np.zeros(self.base_dem_array.shape)
                else:
                    t_surf = self.layer_temperatures[0]
                # distributed t_surf is now added into self.params for longwave flux computation:
                self.aws = AwsParams(float(row["T_AIR"]), float(row["WIND_SPEED"]), float(row["PRESSURE"]), r_hum,
                                     cld, float(row["SWD"]), t_surf, elev_aws, xy_aws[0], xy_aws[1], z)
                if not solar_only:
                    self.params = DistributedParams(self.aws, self.base_dem_array, self.current_date_str, False)
                    # False is for not exporting PNGs

                # interpolating albedo map for the current date:
                self.albedo = interpolate_array(self.albedo_arrays, self.current_date_str)
                # albedo of snow can't be lower than 0.65:
                self.albedo = np.where((self.swe_array > 0) & (self.albedo < 0.65), 0.65, self.albedo)
                # albedo of ice can't be higher than 0.50:
                self.albedo = np.where((self.swe_array <= 0) & (self.albedo > 0.50), 0.50, self.albedo)
                """
                # DEBUG: constant ice and snow albedo:
                self.albedo = np.where(self.swe_array > 0, 0.80, self.albedo)
                self.albedo = np.where(self.swe_array <= 0, 0.38, self.albedo)
                """
                melt_flux = self.calc_heat_fluxes(time_step, solar_only=solar_only, png=png, v=v)
                # print("Melt flux: %.2f W m-2" % np.nanmean(melt_flux))

                snow_melt_amount_we, ice_melt_amount_we = calc_melt(melt_flux, self.swe_array, time_step)
                mean_snow_melt = float(np.nanmean(snow_melt_amount_we))
                mean_ice_melt = float(np.nanmean(ice_melt_amount_we))
                mean_swe = float(np.nanmean(self.swe_array))
                print("Snow/ice melt amount / snow remaining: %.5f/%.5f/%.5f m w.e." % (
                    mean_snow_melt, mean_ice_melt, mean_swe))

                # snow cover percent:
                snow_px = np.sum(self.swe_array > 0)
                total_px = np.count_nonzero(~np.isnan(self.swe_array))
                snow_cover_percent = round(snow_px / total_px * 100)
                print("Percent of snow cover:", snow_cover_percent)

                # then we should update snow reservoir:
                self.swe_array -= snow_melt_amount_we
                # ^^^ no need to check if the new swe_array is below zero: calc_melt function did this already
                self.total_snow_melt_array += snow_melt_amount_we
                self.total_ice_melt_array += ice_melt_amount_we

                stats = "%s,%.4f,%.4f,%.4f,%.0f" % (
                    str(self.output_row), mean_snow_melt, mean_ice_melt, mean_swe, snow_cover_percent)
                with open(out_file, "a") as output:
                    output.write("\n%s" % stats)

                if png:
                    show_me(self.albedo, title="%s albedo" % self.current_date_str, dir="Albedo")
                    show_me(self.swe_array, title="%s snow remnant, m w.e." % self.current_date_str, dir="Snow remnant")
                    show_me(self.total_ice_melt_array, title="%s total ice ONLY melt, m w.e." % self.current_date_str,
                            dir="Melt amount")
                    show_me(self.total_snow_melt_array, title="%s total snow ONLY melt, m w.e." % self.current_date_str,
                            dir="Melt amount")
                """  # fancy progress bar:
                if len(self.input_list) > 100:
                    if i % (round(len(self.input_list) / 100)) == 0:
                        progress = i * 100 / len(self.input_list)
                        s = "%.0f percent done: |" % progress
                        s += "#" * round(progress / 2) + "_" * (50 - round(progress / 2)) + "|"
                        print(s)
                """
                if self.result_export_dates is not None:
                    if self.current_date_str in self.result_export_dates:
                        self.export_result()

            # the final result is always exported despite the "v" and "png" options:
            self.export_result()

    def export_result(self):
        arrays = (self.total_ice_melt_array, self.total_snow_melt_array, self.swe_array)
        titles = ("total_melt_ice", "total_melt_snow", "remaining_snow_cover")
        for arr, title in zip(arrays, titles):
            show_me(arr, title="%s %s" % (self.current_date_str, title), units="m w.e.", dir="Melt amount")
            export_array_as_geotiff(arr, self.geotransform, self.projection,
                                    os.path.join(OUT_DIR, "%s %s.tiff" % (self.current_date_str, title)))
        print("Result saved as GeoTIFF")

    @staticmethod
    def get_time_step(time_list, i, pattern):
        if i < len(time_list) - 1:
            time_step = datetime.strptime(time_list[i + 1]["DATE"], pattern) - datetime.strptime(time_list[i]["DATE"],
                                                                                                 pattern)
        else:
            time_step = datetime.strptime(time_list[i]["DATE"], pattern) - datetime.strptime(time_list[i - 1]["DATE"],
                                                                                             pattern)
        time_step = int(time_step.total_seconds())
        return time_step

    @staticmethod
    def heuristic_unit_guesser(value, scale=10):
        """
        Converts value in percent to a 0-1 range (scale=100)
        or cloudiness in a range from 0 to 10 to a 0-1 range (scale=10)
        :param value:
        :param scale:
        :return:
        """
        if 1 < value <= scale:
            return value / scale
        elif value <= 1:
            return value
        else:
            raise ValueError("Wrong value encountered")

    def calc_heat_fluxes(self, time_step_seconds, solar_only=False, png=True, v=True):
        if not solar_only:
            # TURBULENT HEAT FLUXES
            # at the AWS (needed to know Monin-Obukhov length L), non-distributed:
            aws = self.aws
            # for iterative L calculation let's use the surface temperature at the heat-balance mast:
            point_t_surf = get_value_by_real_coords(self.layer_temperatures[0], self.geotransform, 478283.5, 8655893.3)
            print("Point t_surf is %s degree C" % round(point_t_surf, 2))
            point_t_surf += 273.15  # turbo equations use the thermodynamic temperature

            sensible_flux, latent_flux, monin_obukhov_length = calc_turbulent_fluxes(aws.z, aws.wind_speed, aws.Tz,
                                                                                     aws.P,
                                                                                     aws.rel_humidity,
                                                                                     surface_temp=point_t_surf)

            # DEBUG: let's try to compute the distributed L length!
            """
            params = self.params
            sensible_flux_array, latent_flux_array, monin_obukhov_length = calc_turbulent_fluxes(aws.z,
                                                                                                 params.wind_speed,
                                                                                                 params.Tz, params.P,
                                                                                                 params.rel_humidity,
                                                                                                 surface_temp=
                                                                                                 self.layer_temperatures[
                                                                                                     0] + 273.15)
            """
            # print(sensible_flux, latent_flux, monin_obukhov_length)
            # at the whole glacier surface, distributed:
            params = self.params
            sensible_flux_array, latent_flux_array, monin_obukhov_length = calc_turbulent_fluxes(aws.z,
                                                                                                 params.wind_speed,
                                                                                                 params.Tz, params.P,
                                                                                                 params.rel_humidity,
                                                                                                 L=monin_obukhov_length,
                                                                                                 surface_temp=
                                                                                                 self.layer_temperatures[
                                                                                                     0] + 273.15)

            # export_array_as_geotiff(sensible_flux_array, self.geotransform, self.projection, "/home/tepex/PycharmProjects/energy/gtiff/sensible.tiff")
            # export_array_as_geotiff(latent_flux_array, self.geotransform, self.projection, "/home/tepex/PycharmProjects/energy/gtiff/latent.tiff")

            # LONGWAVE RADIATION FLUX
            lwd, lwu = self.calc_longwave()
            rl = lwd - lwu

            # exporting preview images for the current time step:
            if png:
                show_me(sensible_flux_array, title="%s Sensible heat flux" % self.current_date_str, units="W m-2",
                        dir="Turbulent fluxes")
                show_me(latent_flux_array, title="%s Latent heat flux" % self.current_date_str, units="W m-2",
                        dir="Turbulent fluxes")
                show_me(rl, title="%s Longwave balance" % self.current_date_str, units="W m-2", dir="Fluxes")
        else:
            lwd = 0
            lwu = 0
            sensible_flux_array = 0
            latent_flux_array = 0
            point_t_surf = 273.15

        # SHORTWAVE RADIATION FLUX
        rs = self.calc_shortwave(time_step=time_step_seconds, png=png, v=v)

        atmo_flux = rs + lwd - lwu + sensible_flux_array + latent_flux_array

        if self.use_msm:
            if png:
                for i in range(len(self.layer_temperatures)):  # temperatures BEFORE the update
                    show_me(self.layer_temperatures[i], title="%s Layer %s temperature" % (self.current_date_str, i),
                            units="degree Celsius", dir="Glacier body temperature")
            ######### DEBUG:
            out_temp_str = "\n%s" % self.current_date_str
            for i in range(len(self.layer_temperatures)):
                point_temp = get_value_by_real_coords(self.layer_temperatures[i], self.geotransform, 478283.5,
                                                      8655893.3)  # these ae the real observation coords
                out_temp_str += ",%.2f" % point_temp
            with open(os.path.join(OUT_DIR, "subsurface_temperature_output.csv"), "a") as f:
                f.write(out_temp_str)
            ################
            snow_depth = self.swe_array / CONST["snow_density"]
            updated_temps, melt_flux, g_flux = tick(self.layer_depths, self.layer_temperatures, time_step_seconds,
                                                    flux=atmo_flux, snow_depth=snow_depth)
            self.layer_temperatures = updated_temps
        else:
            # if we don't use the MSM simulation, just assume the in-glacier heat flux negligible:
            g_flux = np.zeros(atmo_flux.shape)
            # since the surface temperature is assumed to be zero degree Celsius,
            # all the heat supply from the atmosphere is available for ice/snow melt:
            melt_flux = atmo_flux + g_flux
            melt_flux[melt_flux < 0] = 0

        self.output_row = OutputRow(self.current_date_str, lwd, lwu, rs, sensible_flux_array, latent_flux_array, atmo_flux, g_flux,
                        melt_flux, point_t_surf - 273.15)

        if png:
            show_me(rs, title="%s Incoming shortwave * (1 - albedo)" % self.current_date_str, units="W m-2",
                    dir="Fluxes")
            show_me(melt_flux, title="%s Heat available for melt" % self.current_date_str, units="W m-2", dir="Fluxes")
            show_me(g_flux, title="%s In-glacier heat flux" % self.current_date_str, units="W m-2", dir="Fluxes")
            show_me(atmo_flux, title="%s Atmospheric heat flux" % self.current_date_str, units="W m-2", dir="Fluxes")

        return melt_flux

    def calc_shortwave(self, time_step, png=True, v=True):
        if self.use_precomputed:
            self.potential_incoming_sr_path = os.path.join(os.path.dirname(self.base_dem_path),
                                                           "%s_total.sdat" % self.current_date_str)
            print("Reading insolation from %s" % self.potential_incoming_sr_path)
        else:
            self.potential_incoming_sr_path = simulate_lighting(self.base_dem_path, self.current_date_str,
                                                                time_step=time_step)
        self.potential_incoming_radiation = load_raster(self.potential_incoming_sr_path, self.outlines_path, v=v)[0]
        self.incoming_shortwave = self.J_to_W(self.kWh_to_J(self.potential_incoming_radiation), time_step=time_step)
        if png:
            show_me(self.incoming_shortwave, title="%s Potential Incoming Solar Radiation" % self.current_date_str,
                    units="W m-2", dir="Fluxes")

        # potential incoming solar radiation into real:
        self.incoming_shortwave *= self.potential_to_real_insolation_factor(time_step, v=v)
        if png:
            show_me(self.incoming_shortwave, title="%s Real incoming solar radiation" % self.current_date_str,
                    units="W m-2", dir="Fluxes")

        if (not self.export_potential) and (not self.use_precomputed):
            cleanup_sgrd(self.potential_incoming_sr_path)

        return self.incoming_shortwave * (1 - self.albedo)

    def potential_to_real_insolation_factor(self, time_step, v=True):
        """
        SUPER-DUPER-PRECISE empirical (NO) coefficient
        :return:
        """
        # should compute factor based on a self.incoming_shortwave_aws and
        # on the corresponding pixel of self.incoming_shortwave
        query = 'gdallocationinfo -valonly -geoloc "%s" %s %s' % (
            self.potential_incoming_sr_path, self.aws.x, self.aws.y)
        potential_at_aws = float(os.popen(query).read())
        potential_at_aws = self.J_to_W(self.kWh_to_J(potential_at_aws), time_step=time_step)

        real_at_aws = self.aws.incoming_shortwave

        ##### DEBUG:   ###########################################
        with open(os.path.join(OUT_DIR, "solar_output.csv"), "a") as f:
            f.write("\n%s,%s,%s" % (self.current_date_str, potential_at_aws, real_at_aws))
        ##########################################################

        if potential_at_aws == 0:
            factor = 1  # makes the real insolation = 0 everywhere
        else:
            factor = real_at_aws / potential_at_aws
        if v:
            print("Potential/observed/scaling factor for incoming solar radiation at AWS location is %.1f/%.1f/%.3f" %
                  (potential_at_aws, real_at_aws, factor))
        return factor

    @staticmethod
    def calc_melt_amount(ice_heat_rate, time_step=None):
        """
        DEPRECATED - was used for ice ablation only. Now we separate snow and ice melt.
        Computes a melt ice layer [m w.e.]
        :param time_step:
        :param ice_heat_rate:
        :return: thickness of melt ice layer in meters w.e.
        """
        if time_step is None:
            time_step = 86400  # seconds per one day
        ice_melt = ice_heat_rate * time_step

        if type(ice_melt) == np.ndarray:
            ice_melt[ice_melt < 0] = 0  # since negative ice melt (i.e. ice accumulation) is not possible
        else:
            ice_melt = 0 if ice_melt < 0 else ice_melt

        return ice_melt

    def calc_longwave(self):
        sigma = 5.70 * 10 ** -8  # Stefan–Boltzmann constant
        eps = 0.98  # thermal emissivity of glacier ice
        lwu = eps * sigma * self.params.Tz_surf ** 4
        lwd = (0.765 + 0.22 * self.aws.cloudiness ** 3) * sigma * self.params.Tz ** 4
        return lwd, lwu

    @staticmethod
    def kWh_to_J(insol):
        """
        Converts amount of energy in [kW*h] into [J]
        :param insol:
        :return:
        """
        return insol * 3.6 * 10 ** 6

    @staticmethod
    def J_to_W(insol, time_step=None):
        """
        Converts amount of energy in [J/period-of-time-in-seconds] into the flux in [W/s]
        :param insol: radiation amount in [J]
        :param time_step: period of time in seconds (one day by default)
        :return: radiation flux in [W/s]
        """
        if time_step is None:
            time_step = 86400
        return insol / time_step

    @staticmethod
    def fill_header(out_file):
        with open(out_file, "w") as output:
            output.write("# DATE format is %Y%m%d, HEAT FLUXES are in W m-2")
            output.write("# ICE and SNOW_MELT are in m w.e.")
            output.write("\n# POINT_T_SURF (degree Celsius) is near the point of glacier body temperature measurements")
            output.write(
                "\nDATE,RS_BALANCE,RL_BALANCE,LWD_FLUX,SENSIBLE,LATENT,ATMO_BALANCE,INSIDE_GLACIER_FLUX,MELT_FLUX,POINT_T_SURF,SNOW_MELT,ICE_MELT,SNOW_COVER,SNOW_COVER_PERCENT_FROM_SURFACE")

    @staticmethod
    def read_input_file(input_file):
        with open(input_file) as csvfile:
            reader = csv.DictReader(csvfile)
            return list(reader)


if __name__ == "__main__":
    arcticdem_path = "/home/tepex/AARI/Glaciology_2019/lighting/source/dem.tif"
    outlines_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/shp/aldegonda_outlines_2018/aldegonda_outlines_2018.shp"
    aws_file = "/home/tepex/PycharmProjects/energy/aws/test_aws_data.csv"
    albedo_maps = {
        "20190727": "/home/tepex/AARI/Glaciers/Aldegonda albedo/gridded/L8_albedo_20190727_crop.tif",
        "20190803": "/home/tepex/AARI/Glaciers/Aldegonda albedo/gridded/L8_albedo_20190803_crop.tif"
    }
    elev_aws = 290
    xy_aws = (478342, 8655635)  # EPSG:32633
    out_file = "/home/tepex/PycharmProjects/energy/aws/test_out.csv"
    #
    e = Energy(arcticdem_path, outlines_path)
    e.model(aws_file=aws_file, out_file=out_file, albedo_maps=albedo_maps, z=1.6, elev_aws=elev_aws, xy_aws=xy_aws)
