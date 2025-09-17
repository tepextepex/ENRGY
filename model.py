import os
import numpy as np
from datetime import datetime
from math import exp
from timeit import timeit
from helpers import J_to_W, fill_header, read_input_file, kWh_to_J, get_time_step, heuristic_unit_guesser
from var_classes import PARAMS
from var_classes import AwsVars, DistributedVars, OutputRow
from turbo import calc_turbulent_fluxes
from raster_utils import show_me, load_raster, export_array_as_geotiff, get_value_by_real_coords
from interpolator import interpolate_array
from saga_lighting import simulate_lighting, cleanup_sgrd
from msm import calc_melt, tick

import pandas as pd


class Energy:
    def __init__(self, base_dem_path, glacier_outlines_path, out_dir, res=None):

        self.params = PARAMS

        self.current_date_str = None
        self.input_list = []
        self.output_row = None

        self.debug_point_output = None

        # spatial resolution in meters:
        if res is None:
            self.res = 100
        else:
            self.res = res

        if os.path.isdir(out_dir):
            self.out_dir = out_dir
        else:
            os.mkdir(out_dir)
            self.out_dir = out_dir
        # self.out_dir = os.path.dirname(os.path.realpath(__file__))

        self.png_export = 1  # will export every n-th png, default is exporting on each time step
        self.result_export_dates = None

        # for potential incoming solar radiation:
        self.export_potential = False
        self.use_precomputed = False

        self.aws = None
        self.vars = None
        self.albedo = None
        self.albedo_arrays = None

        self.cloud_corr = None
        self.sensible_corr_factor = 1
        self.latent_corr_factor = 1

        self.stake_df = None
        self.out_stake_df = None
        self.use_msm = False
        self.msm_xy = None
        self.layer_temperatures = None
        self.layer_depths = []

        self.potential_incoming_sr_path = None
        self.potential_incoming_radiation = None
        self.incoming_shortwave = None
        self.pickle_dir = None

        self.base_dem_path = base_dem_path
        self.outlines_path = glacier_outlines_path

        print("Loading base DEM...")
        self.base_dem_array, self.geotransform, self.projection = load_raster(base_dem_path, self.outlines_path,
                                                                              self.res, v=False)
        self.total_snow_melt_array = np.zeros_like(self.base_dem_array, dtype=np.float32)
        self.total_ice_melt_array = np.zeros_like(self.base_dem_array, dtype=np.float32)

        self.swe_array = np.zeros_like(self.base_dem_array,
                                       dtype=np.float32)  # snow cover may be added later using add_snow method
        # DEBUG:
        # self.swe_array.fill(0.005)

    def set_density(self, snow=None, ice=None):
        if snow is not None:
            self.params["snow_density"] = snow
        if ice is not None:
            self.params["ice_density"] = ice

    def add_cloud_corr(self, cloud_corr):
        if (float(cloud_corr) < -1.0) or (float(cloud_corr) > 1.0):
            raise ValueError("cloud_corr value should be a float between [-1.0..+1.0]")
        else:
            self.cloud_corr = cloud_corr

    def add_pickle_dir(self, pickle_dir):
        self.pickle_dir = os.path.join(pickle_dir, str(self.res))
        if not os.path.exists(self.pickle_dir):
            raise IOError(f"Cannot find pickled insolation for {self.res} m resolution inside {pickle_dir}!"
                          f"Please choose directory containing pickles or change the spatial resolution.")

    def add_stakes(self, file_path):
        self.stake_df = pd.read_csv(file_path)
        self.out_stake_df = pd.DataFrame({"name": self.stake_df["name"]})

    def write_stakes(self, out_file_path):
        out_path = os.path.join(os.path.dirname(out_file_path), "ice_melt_point.csv")
        # print(out_path)
        self.out_stake_df.to_csv(out_path, index=False, float_format="%.3f")

    def sample_stakes(self):
        vals = []
        for row in self.stake_df.itertuples():
            try:
                value = get_value_by_real_coords(self.total_ice_melt_array, self.geotransform, row.easting, row.northing)
                value = round(value, 4)
            except Exception as e:
                value = None
            vals.append(value)
        self.out_stake_df[self.current_date_str] = vals

    def add_snow(self, swe_map_path):
        print("Initialized snow cover state (SWE) from %s" % swe_map_path)
        self.swe_array = load_raster(swe_map_path, self.outlines_path, self.res, v=False)[0]

    def add_msm(self, depths, temperatures, elev_aws):
        print("Initializing subsurface model...")

        self.use_msm = True
        self.layer_depths = depths
        self.layer_temperatures = []

        def extrapolate(value, v_gradient=None):
            delta_dem = self.base_dem_array - elev_aws
            if v_gradient is None:
                # v_gradient = -0.008  # 8 degrees Celsius or Kelvins per 1 km, year 2021 value
                v_gradient = -0.006
            return value + delta_dem * v_gradient

        for t_point in temperatures:
            t_distributed = extrapolate(t_point)
            t_distributed[t_distributed > 0] = 0.0  # ice temperature is limited by melting point
            self.layer_temperatures.append(t_distributed)
        # print(self.layer_depths)

        i = 0
        for temp_array in self.layer_temperatures:
            show_me(temp_array, self.out_dir, title="Layer %s temperature" % i, dir="Glacier body temperature")
            i += 1

    def add_checkpoints(self, date_str_list):
        dts = [s + " 12:00:00" for s in date_str_list]
        self.result_export_dates = dts

    def model(self, aws_file=None, albedo_maps=None, z=2.0, elev_aws=0.0, xy_aws=None,
              zm=None, z_h_or_e=None, andreas=False,
              solar_only=False, const_albedo=None, temp_lapse_rate=-0.006, last_snowfall=None,
              max_ice_albedo=None, emissivity=None, v=True):
        if aws_file is not None:
            if albedo_maps is not None:
                # loading albedo maps from geotiff files into arrays:
                self.albedo_arrays = {}
                for key in albedo_maps:  # albedo_maps contains file paths
                    self.albedo_arrays[key] = load_raster(albedo_maps[key], self.outlines_path, self.res,
                                                          remove_outliers=True, v=v)[0]

            out_file = os.path.join(self.out_dir, "heat_fluxes.csv")
            fill_header(out_file)

            if self.debug_point_output is not None:
                header = ""
                with open(os.path.join(self.out_dir, self.debug_point_output), "a") as f:
                    if self.use_msm:
                        cur_depth = 0.0  # surface layer, zero depth
                        header += f"{cur_depth},"
                        for layer_thickness in self.layer_depths:  # yes, in fact these are layer THICKNESSES, not the depths below the suface
                            cur_depth += layer_thickness
                            header += f"{cur_depth},"
                    header += "SENSIBLE,LATENT"
                    f.write(header)

            self.input_list = read_input_file(aws_file)
            for i in range(len(self.input_list)):
                png = True if i % self.png_export == 0 else False

                row = self.input_list[i]
                print("\nProcessing %s..." % row["DATE"])
                self.current_date_str = row["DATE"]

                try:
                    time_step = get_time_step(self.input_list, i, "%Y%m%d")
                except ValueError:
                    time_step = get_time_step(self.input_list, i, "%Y%m%d %H:%M:%S")
                # print("########## TIME STEP IS: %s" % time_step)  # DEBUG

                # setting meteo parameters for the current date:
                r_hum = heuristic_unit_guesser(float(row["HUMID"]), 100)

                # Cloud coverage:
                cld = float(row["CLOUDINESS"])
                if self.cloud_corr is not None:
                    cld += self.cloud_corr
                    cld = 1.0 if cld > 1.0 else cld
                    cld = 0.0 if cld < 0.0 else cld
                    # print(cld)  # DEBUG

                if self.layer_temperatures is None:
                    t_surf = np.zeros(self.base_dem_array.shape)
                else:
                    t_surf = self.layer_temperatures[0]

                # setting the air temperature vertical lapse for inter- and extrapolation:
                try:
                    grad_temp = float(temp_lapse_rate)  # temp_lapse_rate could be a constant float value
                except ValueError:
                    try:
                        grad_temp = float(row["GRADIENT"])  # it also could be a field name (string)
                    except KeyError:
                        # if this field does not exist, then setting default:
                        print(f"Setting default value of {temp_lapse_rate} for the air temperature lapse")
                        grad_temp = temp_lapse_rate  # 6 degrees Celsius or Kelvins per 1 km
                finally:
                    if v:
                        print(f"Lapse rate for the t_air set to {grad_temp}")
                    else:
                        pass

                # distributed t_surf is now added into self.vars for longwave flux computation:
                self.aws = AwsVars(float(row["T_AIR"]), float(row["WIND_SPEED"]), float(row["PRESSURE"]), r_hum,
                                   cld, float(row["SWD"]), t_surf, grad_temp, elev_aws, xy_aws[0], xy_aws[1], z)
                if not solar_only:
                    self.vars = DistributedVars(self.aws, self.base_dem_array, self.current_date_str, False)
                    # False is for not exporting PNGs

                self.albedo = self.calc_albedo(constant=const_albedo, last_snowfall=last_snowfall,
                                               max_ice_albedo=max_ice_albedo, v=v)

                melt_flux = self.calc_energy_fluxes(time_step, zm=zm, z_h_or_e=z_h_or_e,
                                                    andreas=andreas,
                                                    emissivity=emissivity, xy_aws=xy_aws,
                                                    solar_only=solar_only, png=png, v=v)
                # print("Melt flux: %.2f W m-2" % np.nanmean(melt_flux))

                if not solar_only:
                    snow_melt_we, ice_melt_we = calc_melt(melt_flux, self.swe_array, time_step)
                    mean_snow_melt = float(np.nanmean(snow_melt_we))
                    mean_ice_melt = float(np.nanmean(ice_melt_we))
                    mean_swe = float(np.nanmean(self.swe_array))
                    # snow cover percent:
                    snow_px = np.sum(self.swe_array > 0)
                    total_px = np.count_nonzero(~np.isnan(self.swe_array))
                    snow_cover_percent = round(snow_px / total_px * 100)
                    if v:
                        print("Snow/ice melt amount / snow remaining / percent of snow cover:\n%.3f/%.3f/%.3f m w.e./%s" % (
                            mean_snow_melt, mean_ice_melt, mean_swe, snow_cover_percent))

                    # then we should update snow reservoir:
                    self.swe_array -= snow_melt_we
                    # ^^^ no need to check if the new swe_array is below zero: calc_melt function did this already
                    self.total_snow_melt_array += snow_melt_we
                    self.total_ice_melt_array += ice_melt_we
                else:
                    pass
                    mean_snow_melt, mean_ice_melt, mean_swe, snow_cover_percent = 0, 0, 0, 0

                stats = "%s,%.4f,%.4f,%.4f,%.0f" % (
                    str(self.output_row), mean_snow_melt, mean_ice_melt, mean_swe, snow_cover_percent)
                with open(out_file, "a") as output:
                    output.write("\n%s" % stats)

                if png:
                    show_me(self.albedo, self.out_dir, title="%s albedo" % self.current_date_str, dir="Albedo")
                    show_me(self.swe_array, self.out_dir, title="%s snow remnant, m w.e." % self.current_date_str, dir="Snow remnant")
                    show_me(self.total_ice_melt_array, self.out_dir, title="%s total ice ONLY melt, m w.e." % self.current_date_str,
                            dir="Melt amount")
                    show_me(self.total_snow_melt_array, self.out_dir, title="%s total snow ONLY melt, m w.e." % self.current_date_str,
                            dir="Melt amount")

                if self.result_export_dates is not None:
                    if self.current_date_str in self.result_export_dates:
                        self.export_result()
                        self.sample_stakes()
                        self.write_stakes(out_file)

            # the final result is always exported despite the "v" and "png" options:
            self.export_result()

    def export_result(self):
        arrays = (self.total_ice_melt_array, self.total_snow_melt_array, self.swe_array)
        titles = ("total_melt_ice", "total_melt_snow", "remaining_snow_cover")
        for arr, title in zip(arrays, titles):
            show_me(arr, self.out_dir, title="%s %s" % (self.current_date_str, title), units="m w.e.", dir="Melt amount")
            export_array_as_geotiff(arr, self.geotransform, self.projection,
                                    os.path.join(self.out_dir, "%s %s.tiff" % (self.current_date_str, title)))
        print("Result saved as GeoTIFF")

    #@timeit
    def calc_albedo(self, constant=None, last_snowfall=None, max_ice_albedo=None, v=False):
        """

        :param max_ice_albedo:
        :param last_snowfall: the date after which the snow albedo will decrease ("%Y%m%d")
        :param constant: tuple of floats (ice_albedo, snow_albedo)
        :return:
        """
        if constant is None:
            # interpolating albedo map for the current date:
            a = interpolate_array(self.albedo_arrays, self.current_date_str)
            # albedo of snow can't be lower than 0.65:
            # a = np.where((self.swe_array > 0) & (a < 0.65), 0.65, a)
            if last_snowfall is not None:
                # let's force the snow albedo decrease in time:
                # delta = datetime.strptime(self.current_date_str, "%Y%m%d %H:%M:%S") - datetime.strptime("20220601", "%Y%m%d")  # for the year 2022
                delta = (datetime.strptime(self.current_date_str, "%Y%m%d %H:%M:%S")
                         - datetime.strptime(last_snowfall, "%Y%m%d"))
                # delta = datetime.strptime(self.current_date_str, "%Y%m%d %H:%M:%S") - datetime.strptime("20210611", "%Y%m%d")  # for the real 2021 data
                delta_days = delta.days
                if delta_days > 0:
                    snow_albedo = 0.40 + 0.44 * exp(-0.12 * delta_days)
                    a = np.where(self.swe_array > 0, snow_albedo, a)
                    if v:
                        print("Setting snow albedo to %.2f" % snow_albedo)

            # albedo of ice can't be higher than a threshold value:
            if max_ice_albedo is None:
                max_ice_albedo = 0.45
            a = np.where((self.swe_array <= 0) & (a > max_ice_albedo), max_ice_albedo, a)
        else:
            # constant ice and snow albedo:
            snow_albedo = constant[1]
            ice_albedo = constant[0]
            a = np.where(self.swe_array > 0, snow_albedo, ice_albedo)

            # TODO: what if swe_array was not initialized ??

        # return a - 0.1
        return a

    #@timeit
    def calc_energy_fluxes(self, time_step_seconds, emissivity=None, xy_aws=None,
                           zm=None, z_h_or_e=None, andreas=False, solar_only=False, png=True, v=True):
        if not solar_only:
            # TURBULENT HEAT FLUXES
            # at the AWS (needed to know Monin-Obukhov length L), non-distributed:
            aws = self.aws
            # for iterative L calculation let's use the surface temperature at the heat-balance mast:
            point_t_surf = get_value_by_real_coords(self.layer_temperatures[0], self.geotransform, *xy_aws)
            if v:
                print("Point t_surf is %s degree C" % round(point_t_surf, 2))
            point_t_surf += 273.15  # turbo equations use thermodynamic temperature in [K]

            # first we'll solve the equation at one point to obtain the monin-obukhov length:
            sensible_flux, latent_flux, monin_obukhov_length = calc_turbulent_fluxes(aws.z, aws.wind_speed, aws.Tz,
                                                                                     aws.P,
                                                                                     aws.rel_humidity,
                                                                                     zm=zm, z_h_or_e=z_h_or_e,
                                                                                     andreas=andreas,
                                                                                     surface_temp=point_t_surf)

            # DEBUG: let's try to compute the distributed L length!
            """
            sensible_flux_array, latent_flux_array, monin_obukhov_length = calc_turbulent_fluxes(aws.z,
                                                                                                 self.vars.wind_speed,
                                                                                                 self.vars.Tz, self.vars.P,
                                                                                                 self.vars.rel_humidity,
                                                                                                 surface_temp=
                                                                                                 self.layer_temperatures[
                                                                                                     0] + 273.15)
            """
            # print(sensible_flux, latent_flux, monin_obukhov_length)
            # at the whole glacier surface, distributed:
            sensible_flux_array, latent_flux_array, monin_obukhov_length = calc_turbulent_fluxes(aws.z,
                                                                                                 self.vars.wind_speed,
                                                                                                 self.vars.Tz, self.vars.P,
                                                                                                 self.vars.rel_humidity,
                                                                                                 L=monin_obukhov_length,
                                                                                                 zm=zm, z_h_or_e=z_h_or_e,
                                                                                                 andreas=andreas,
                                                                                                 surface_temp=
                                                                                                 self.layer_temperatures[
                                                                                                     0] + 273.15)
            # DEBUG:
            # export_array_as_geotiff(sensible_flux_array, self.geotransform, self.projection, os.path.join(self.out_dir, "sensible.tiff"))
            # export_array_as_geotiff(latent_flux_array, self.geotransform, self.projection, os.path.join(self.out_dir, "latent.tiff"))

            sensible_flux_array = sensible_flux_array * self.sensible_corr_factor
            latent_flux_array = latent_flux_array * self.latent_corr_factor

            # LONGWAVE RADIATION FLUX
            lwd, lwu = self.calc_longwave(eps=emissivity)
            rl = lwd - lwu

            # exporting preview images for the current time step:
            if png:
                show_me(sensible_flux_array, self.out_dir, title="%s Sensible heat flux" % self.current_date_str, units="W m-2",
                        dir="Turbulent fluxes")
                show_me(latent_flux_array, self.out_dir, title="%s Latent heat flux" % self.current_date_str, units="W m-2",
                        dir="Turbulent fluxes")
                show_me(rl, self.out_dir, title="%s Longwave balance" % self.current_date_str, units="W m-2", dir="Fluxes")
        else:
            lwd = 0
            lwu = 0
            sensible_flux_array = 0
            latent_flux_array = 0
            point_t_surf = 273.15

        # SHORTWAVE RADIATION FLUX
        rs = self.calc_shortwave(time_step=time_step_seconds, png=png, v=v)

        # TOTAL ATMOSPHERIC FLUX
        atmo_flux = rs + lwd - lwu + sensible_flux_array + latent_flux_array

        out_point_str = "\n%s" % self.current_date_str  # Used later for debug_point_output, if enabled

        if self.use_msm:
            if png:
                for i in range(len(self.layer_temperatures)):  # temperatures BEFORE the update
                    show_me(self.layer_temperatures[i], self.out_dir, title="%s Layer %s temperature" % (self.current_date_str, i),
                            units="degree Celsius", dir="Glacier body temperature")

            if self.msm_xy is not None:
                for i in range(len(self.layer_temperatures)):
                    point_temp = get_value_by_real_coords(self.layer_temperatures[i],
                                                          self.geotransform,
                                                          *self.msm_xy)
                    out_point_str += ",%.2f" % point_temp

            snow_depth = self.swe_array / PARAMS["snow_density"]
            updated_temps, melt_flux, g_flux = tick(self.layer_depths, self.layer_temperatures, time_step_seconds,
                                                    flux=atmo_flux, snow_depth=snow_depth)
            self.layer_temperatures = updated_temps
        else:
            # if we don't use the MSM simulation, just assume the in-glacier heat flux negligible:
            g_flux = np.zeros(atmo_flux.shape)
            # since the surface temperature is assumed to be zero degree Celsius,
            # all the energy supply from the atmosphere is available for ice/snow melt:
            melt_flux = atmo_flux + g_flux
            melt_flux[melt_flux < 0] = 0

        # debug output, temperatures and turbulent fluxes at their measurement locations:
        if self.debug_point_output is not None:
            # getting the turbulent fluxes at the AWS installation site:
            point_sensible = get_value_by_real_coords(sensible_flux_array, self.geotransform, *xy_aws)
            point_latent = get_value_by_real_coords(latent_flux_array, self.geotransform, *xy_aws)
            out_point_str += ",%.1f,%.1f" % (point_sensible, point_latent)

            with open(os.path.join(self.out_dir, self.debug_point_output), "a") as f:
                f.write(out_point_str)

        # regular model output, area-averaged values:
        self.output_row = OutputRow(self.current_date_str, lwd, lwu, rs, sensible_flux_array, latent_flux_array,
                                    atmo_flux, g_flux, melt_flux, point_t_surf - 273.15)

        if png:
            show_me(rs, self.out_dir, title="%s Incoming shortwave * (1 - albedo)" % self.current_date_str, units="W m-2",
                    dir="Fluxes")
            show_me(melt_flux, self.out_dir, title="%s Heat available for melt" % self.current_date_str, units="W m-2", dir="Fluxes")
            show_me(g_flux, self.out_dir, title="%s In-glacier heat flux" % self.current_date_str, units="W m-2", dir="Fluxes")
            show_me(atmo_flux, self.out_dir, title="%s Atmospheric heat flux" % self.current_date_str, units="W m-2", dir="Fluxes")

        return melt_flux

    #@timeit
    def calc_shortwave(self, time_step, png=True, v=True):
        if self.use_precomputed:
            self.potential_incoming_sr_path = os.path.join(os.path.dirname(self.base_dem_path),
                                                           "%s_total.sdat" % self.current_date_str)
        else:
            self.potential_incoming_sr_path = simulate_lighting(self.base_dem_path, self.current_date_str,
                                                                time_step=time_step)
        if self.pickle_dir is None:
            if v:
                print(f"Reading insolation from {self.potential_incoming_sr_path}")
            self.potential_incoming_radiation = load_raster(self.potential_incoming_sr_path, self.outlines_path,
                                                            self.res, v=v)[0]
        else:
            pkl_base_name = f"{os.path.basename(self.potential_incoming_sr_path)}.npy"
            pkl_path = os.path.join(self.pickle_dir, pkl_base_name)
            if v:
                print(f"Unpickling insolation from {pkl_path}")
            self.potential_incoming_radiation = np.load(pkl_path)

        self.incoming_shortwave = J_to_W(kWh_to_J(self.potential_incoming_radiation), time_step=time_step)
        if png:
            show_me(self.incoming_shortwave, self.out_dir, title="%s Potential Incoming Solar Radiation" % self.current_date_str,
                    units="W m-2", dir="Fluxes")

        # potential incoming solar radiation into real:
        self.incoming_shortwave *= self.potential_to_real_insolation_factor(time_step, v=v)
        if png:
            show_me(self.incoming_shortwave, self.out_dir, title="%s Real incoming solar radiation" % self.current_date_str,
                    units="W m-2", dir="Fluxes")

        if (not self.export_potential) and (not self.use_precomputed):
            cleanup_sgrd(self.potential_incoming_sr_path)

        return self.incoming_shortwave * (1 - self.albedo)

    #@timeit
    def potential_to_real_insolation_factor(self, time_step, v=True):
        """
        SUPER-DUPER-PRECISE (NO) empirical coefficient
        :return:
        """
        # should compute factor based on a self.incoming_shortwave_aws and
        # on the corresponding pixel of self.incoming_shortwave
        """
        query = 'gdallocationinfo -valonly -geoloc "%s" %s %s' % (
            self.potential_incoming_sr_path, self.aws.x, self.aws.y)
        potential_at_aws = float(os.popen(query).read())  # popen is too slow, don't use (!)
        """
        potential_at_aws = get_value_by_real_coords(self.potential_incoming_radiation, self.geotransform,
                                                    self.aws.x, self.aws.y)
        potential_at_aws = J_to_W(kWh_to_J(potential_at_aws), time_step=time_step)

        real_at_aws = self.aws.incoming_shortwave

        ##### DEBUG:   ###########################################
        with open(os.path.join(self.out_dir, "solar_output.csv"), "a") as f:
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

    #@timeit
    def calc_longwave(self, eps=None):
        """
        Downward longwave radiation is computed using the well-known parametrization for polar regions
        by König-Langlo & Augstein (1994)
        :param eps: thermal emissivity of glacier surface (usually 0.96-1.00)
        :return: downward flux, upward flux
        """
        sigma = 5.70 * 10 ** -8  # Stefan–Boltzmann constant
        if eps is None:
            eps = 0.98
        lwu = eps * sigma * self.vars.Tz_surf ** 4
        lwd = (0.765 + 0.22 * self.aws.cloudiness ** 3) * sigma * self.vars.Tz ** 4
        return lwd, lwu


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
    e = Energy(arcticdem_path, outlines_path, out_dir="/home/tepex/PycharmProjects/energy/2025_potential_incoming")
    e.model(aws_file=aws_file, albedo_maps=albedo_maps, z=1.6, elev_aws=elev_aws, xy_aws=xy_aws)
