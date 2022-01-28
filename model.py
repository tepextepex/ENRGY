import os
import csv
import numpy as np
from datetime import datetime

import msm
from parameter_classes import CONST
from parameter_classes import AwsParams, DistributedParams, OutputRow
from turbo import calc_turbulent_fluxes
from raster_utils import show_me, load_raster, export_array_as_geotiff, OUT_DIR
from interpolator import interpolate_array
from saga_lighting import simulate_lighting, cleanup_sgrd
from msm import calc_melt


class Energy:
    def __init__(self, base_dem_path, glacier_outlines_path):

        self._init_constants()

        self.current_date_str = None
        # self.modelled_days = 0
        self.input_list = []
        self.output_list = []

        self.aws = None
        self.params = None
        self.albedo = None
        self.albedo_arrays = None

        self.potential_incoming_sr_path = None
        self.potential_incoming_radiation = None
        self.incoming_shortwave = None

        self.base_dem_path = base_dem_path
        self.outlines_path = glacier_outlines_path

        print("Loading base DEM...")
        self.base_dem_array, self.geotransform, self.projection = load_raster(base_dem_path, self.outlines_path)
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
        self.swe_array = load_raster(swe_map_path, self.outlines_path)[0]

    def model(self, aws_file=None, out_file=None, albedo_maps=None, z=2.0, elev_aws=0.0, xy_aws=None, solar_only=False,
              png=True, v=True, use_msm=False):
        if (aws_file is not None) and (albedo_maps is not None):
            # loading albedo maps from geotiff files into arrays:
            self.albedo_arrays = {}
            for key in albedo_maps:  # albedo_maps contains file paths
                self.albedo_arrays[key] = load_raster(albedo_maps[key], self.outlines_path, remove_outliers=True, v=v)[
                    0]

            self.fill_header(out_file)

            self.input_list = self.read_input_file(aws_file)
            for i in range(0, len(self.input_list)):
                row = self.input_list[i]
                print("\nProcessing %s..." % row["DATE"])
                self.current_date_str = row["DATE"]

                # setting meteo parameters for the current date:
                r_hum = self.heuristic_unit_guesser(float(row["REL_HUMIDITY"]), 100)
                cld = self.heuristic_unit_guesser(float(row["CLOUDINESS"]), 10)
                self.aws = AwsParams(float(row["T_AIR"]), float(row["WIND_SPEED"]), float(row["AIR_PRESSURE"]), r_hum,
                                     cld, float(row["INCOMING_SHORTWAVE"]), elev_aws, xy_aws[0], xy_aws[1], z)
                if not solar_only:
                    self.params = DistributedParams(self.aws, self.base_dem_array)

                # interpolating albedo map for the current date:
                self.albedo = interpolate_array(self.albedo_arrays, self.current_date_str)
                self.albedo = np.where(self.swe_array > 0, 0.65, self.albedo)
                if png:
                    show_me(self.albedo, title="%s albedo" % self.current_date_str)

                try:
                    time_step = self.get_time_step(self.input_list, i, "%Y%m%d")
                except ValueError:
                    time_step = self.get_time_step(self.input_list, i, "%Y%m%d %H:%M:%S")
                # print("########## TIME STEP IS: %s" % time_step)  # DEBUG

                melt_flux = self.run(time_step, solar_only=solar_only, png=png, v=v, use_msm=use_msm)
                print("Melt flux: %.2f W m-2" % np.nanmean(melt_flux))

                stats = str(self.output_list[-1])
                with open(out_file, "a") as output:
                    output.write("\n%s" % stats)

                snow_melt_amount_we, ice_melt_amount_we = calc_melt(melt_flux, self.swe_array, time_step)
                print("Snow/ice melt amount: %.5f/%.5f m w.e." % (np.nanmean(snow_melt_amount_we), np.nanmean(ice_melt_amount_we)))
                # then we should update snow reservoir:
                self.swe_array -= snow_melt_amount_we
                # ^^^ no need to check if the new swe_array is below zero: calc_melt function did this already
                self.total_snow_melt_array += snow_melt_amount_we
                self.total_ice_melt_array += ice_melt_amount_we

                if png:
                    show_me(self.swe_array, title="%s snow remnant, m w.e." % self.current_date_str)
                    show_me(self.total_ice_melt_array, title="%s total ice ONLY melt, m w.e." % self.current_date_str)

            # the final result is always exported despite the "v" and "png" options:
            for arr, title in zip((self.total_ice_melt_array, self.total_snow_melt_array), ("total_melt_ice", "total_melt_snow")):
                show_me(arr, title="Total melt over the period", units="m w.e.")
                export_array_as_geotiff(arr, self.geotransform, self.projection,
                                        os.path.join(OUT_DIR, "%s.tiff" % title))

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

    def run(self, time_step_seconds, solar_only=False, png=True, v=True, use_msm=False):
        if not solar_only:
            # TURBULENT HEAT FLUXES
            # at the AWS (needed to know Monin-Obukhov length L), non-distributed:
            aws = self.aws
            # TODO: pass the surface temperature into the turbulence formulas
            sensible_flux, latent_flux, monin_obukhov_length = calc_turbulent_fluxes(aws.z, aws.wind_speed, aws.Tz,
                                                                                     aws.P,
                                                                                     aws.rel_humidity)

            # at the whole glacier surface, distributed:
            params = self.params
            sensible_flux_array, latent_flux_array, monin_obukhov_length = calc_turbulent_fluxes(aws.z,
                                                                                                 params.wind_speed,
                                                                                                 params.Tz, params.P,
                                                                                                 params.rel_humidity,
                                                                                                 L=monin_obukhov_length)

            # export_array_as_geotiff(sensible_flux_array, self.geotransform, self.projection, "/home/tepex/PycharmProjects/energy/gtiff/sensible.tiff")
            # export_array_as_geotiff(latent_flux_array, self.geotransform, self.projection, "/home/tepex/PycharmProjects/energy/gtiff/latent.tiff")

            # LONGWAVE RADIATION FLUX
            # TODO: pass the surface temperature into the longwave radiation formulas
            lwd, lwu = self.calc_longwave()
            rl = lwd - lwu

            # exporting preview images for the current time step:
            if png:
                show_me(sensible_flux_array, title="%s Sensible heat flux" % self.current_date_str, units="W m-2")
                show_me(latent_flux_array, title="%s Latent heat flux" % self.current_date_str, units="W m-2")
                show_me(rl, title="%s Longwave" % self.current_date_str, units="W m-2")
        else:
            lwd = 0
            lwu = 0
            sensible_flux_array = 0
            latent_flux_array = 0

        # SHORTWAVE RADIATION FLUX
        rs = self.calc_shortwave(time_step=time_step_seconds, png=png, v=v)

        atmo_flux = rs + lwd - lwu + sensible_flux_array + latent_flux_array
        # TODO: here we should pass atmo_flux into MSM as the atmospheric forcing, and get back the melt_flux
        if use_msm:
            # g_flux = msm.tick()
            pass
        else:
            # if we don't use the MSM simulation, just assume the in-glacier heat flux negligible:
            g_flux = np.zeros(atmo_flux.shape)
        melt_flux = atmo_flux + g_flux
        melt_flux[melt_flux < 0] = 0

        out = OutputRow(self.current_date_str, lwd, lwu, rs, sensible_flux_array, latent_flux_array, atmo_flux, g_flux, melt_flux)
        self.output_list.append(out)

        if png:
            show_me(rs, title="%s Incoming shortwave * (1 - albedo)" % self.current_date_str, units="W m-2")
            show_me(melt_flux, title="%s Heat available for melt" % self.current_date_str, units="W m-2")

        return melt_flux

    def calc_shortwave(self, time_step, png=True, v=True):
        self.potential_incoming_sr_path = simulate_lighting(self.base_dem_path, self.current_date_str,
                                                            time_step=time_step)
        self.potential_incoming_radiation = load_raster(self.potential_incoming_sr_path, self.outlines_path)[0]
        self.incoming_shortwave = self.J_to_W(self.kWh_to_J(self.potential_incoming_radiation), time_step=time_step)
        if png:
            show_me(self.incoming_shortwave, title="%s Potential Incoming Solar Radiation" % self.current_date_str,
                    units="W m-2")

        # potential incoming solar radiation into real:
        self.incoming_shortwave *= self.potential_to_real_insolation_factor(time_step, v=v)
        if png:
            show_me(self.incoming_shortwave, title="%s Real incoming solar radiation" % self.current_date_str,
                    units="W m-2")
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
        lwu = 0.98 * 5.669 * 10 ** -8 * self.params.Tz_surf ** 4
        lwd = (0.765 + 0.22 * self.aws.cloudiness ** 3) * 5.669 * 10 ** -8 * self.params.Tz ** 4
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
            output.write("# DATE format is %Y%m%d, MELT is in m w.e., BALANCES and FLUXES are in W m-2")
            output.write(
                "\nDATE,RS_BALANCE,RL_BALANCE,LWD_FLUX,SENSIBLE,LATENT,ATMO_BALANCE,INSIDE_GLACIER_FLUX,MELT_FLUX")  # heade

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
