import csv
from datetime import datetime

import numpy as np


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


def fill_header(out_file):
    with open(out_file, "w") as output:
        output.write("# DATE format is %Y%m%d, HEAT FLUXES are in W m-2")
        output.write("# ICE and SNOW_MELT are in m w.e.")
        output.write("\n# POINT_T_SURF (degree Celsius) is near the point of glacier body temperature measurements")
        output.write(
            "\nDATE,RS_BALANCE,RL_BALANCE,LWD_FLUX,SENSIBLE,LATENT,ATMO_BALANCE,INSIDE_GLACIER_FLUX,MELT_FLUX,POINT_T_SURF,SNOW_MELT,ICE_MELT,SNOW_COVER,SNOW_COVER_PERCENT_FROM_SURFACE")


def read_input_file(input_file):
    with open(input_file) as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


def kWh_to_J(insol):
    """
    Converts amount of energy in [kW*h] into [J]
    :param insol:
    :return:
    """
    return insol * 3.6 * 10 ** 6


def get_time_step(time_list, i, pattern):
    if i < len(time_list) - 1:
        time_step = datetime.strptime(time_list[i + 1]["DATE"], pattern) - datetime.strptime(time_list[i]["DATE"],
                                                                                             pattern)
    else:
        time_step = datetime.strptime(time_list[i]["DATE"], pattern) - datetime.strptime(time_list[i - 1]["DATE"],
                                                                                         pattern)
    time_step = int(time_step.total_seconds())
    return time_step


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
