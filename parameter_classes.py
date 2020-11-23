import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from turbo import _calc_e_max
from raster_utils import show_me


CONST = {
        "ice_density": 916.7,  # kg m-3
        "latent_heat_of_fusion": 3.34 * 10**5,  # J kg-1
        "g": 9.81
}


@dataclass
class Output:
    date_time_str: str
    lwd: np.array
    lwu: np.array
    rs_balance: np.array
    sensible: np.array
    latent: np.array
    date_time: datetime = field(init=False)
    rl_balance: np.array = field(init=False)
    tr_balance: np.array = field(init=False)
    melt_flux: np.array = field(init=False)
    melt_rate: np.array = field(init=False, metadata={"units": "m w.e. per s"})

    def __post_init__(self):
        self.date_time = datetime.strptime(self.date_time_str, "%Y%m%d")
        #
        self.rl_balance = self.lwd - self.lwu
        self.tr_balance = self.sensible + self.latent
        #
        self.melt_flux = self.rs_balance + self.rl_balance + self.tr_balance
        #
        latent_heat_of_fusion = CONST["latent_heat_of_fusion"]
        ice_density = CONST["ice_density"]
        self.melt_rate = self.melt_flux / (ice_density * latent_heat_of_fusion)
        self.melt_rate[self.melt_rate < 0] = 0  # since negative ice melt is not possible


@dataclass
class AwsParams:
    # meteorological parameters:
    t_air: float = field(metadata={"units": "degree Celsius"})
    wind_speed: float = field(metadata={"units": "m per s"})
    pressure: float = field(metadata={"units": "hPa"})
    rel_humidity: float = field(metadata={"units": "0-1"})
    cloudiness: float = field(metadata={"units": "0-1"})
    incoming_shortwave: float = field(metadata={"units": "W per sq m"})
    # AWS location info:
    elev: float
    x: float
    y: float
    z: float = 2.0
    # post-calculated fields:
    Tz: float = field(init=False, metadata={"units": "Kelvin"})
    P: float = field(init=False, metadata={"units": "Pa"})
    e: float = field(init=False, metadata={"units": "Pa"})  # partial water vapour pressure at the AWS

    def __post_init__(self):
        self.t_surf = 0
        self.Tz = self.t_air + 273.15
        self.P = self.pressure * 100  # Pascals from hPa
        self.e = self.rel_humidity * _calc_e_max(self.Tz, self.P)

    def get_meta(self, field_name):
        return self.__dataclass_fields__[field_name].metadata

    def get_units(self, field_name):
        return self.get_meta(field_name)["units"]


@dataclass
class DistributedParams:
    aws: AwsParams
    dem: np.array = field(metadata={"units": "m", "desc": "Elevation"})
    delta_dem: np.array = field(init=False, metadata={"units": "m", "desc": "Elevation difference relative to AWS location"})  # elevation differences relative to aws location
    t_air: np.array = field(init=False, metadata={"units": "degree Celsius", "desc": "Air temperature"})
    Tz: np.array = field(init=False, metadata={"units": "Kelvin", "desc": "Air thermodynamic temperature"})
    t_surf: np.array = field(init=False, metadata={"units": "degree Celsius", "desc": "Surface temperature"})
    Tz_surf: np.array = field(init=False, metadata={"units": "Kelvin", "desc": "Surface thermodynamic temperature"})
    wind_speed: np.array = field(init=False, metadata={"units": "m per s", "desc": "Wind speed"})
    pressure: np.array = field(init=False, metadata={"units": "hPa", "desc": "Air pressure"})
    e: np.array = field(init=False, metadata={"units": "Pa", "desc": "Partial pressure of water vapour"})
    e_max: np.array = field(init=False, metadata={"units": "Pa", "desc": "Max partial pressure of water vapour"})
    rel_humidity: np.array = field(init=False, metadata={"units": "0-1", "desc": "Relative air humidity"})

    def __post_init__(self):
        self.delta_dem = self.dem - self.aws.elev
        self.t_air = self.__interpolate_t_air()
        self.param_to_png("t_air")
        self.Tz = to_kelvin(self.t_air)
        self.t_surf = self.__fill_array_with_one_value(self.aws.t_surf)
        self.param_to_png("t_surf")
        self.Tz_surf = to_kelvin(self.t_surf)
        self.wind_speed = self.__fill_array_with_one_value(self.aws.wind_speed)
        self.param_to_png("wind_speed")
        self.pressure = self.__interpolate_pressure()
        self.param_to_png("pressure")
        self.P = self.pressure * 100  # Pascals from hPa
        self.e = self.__interpolate_e()
        self.param_to_png("e")
        self.e_max = _calc_e_max(self.Tz, self.P)
        self.param_to_png("e_max")
        self.rel_humidity = np.divide(self.e, self.e_max)  # self.e / self.e_max
        self.param_to_png("rel_humidity")

    def get_meta(self, field_name):
        return self.__dataclass_fields__[field_name].metadata

    def get_units(self, field_name):
        return self.get_meta(field_name)["units"]

    def get_desc(self, field_name):
        return self.get_meta(field_name)["desc"]

    def __interpolate_t_air(self, v_gradient=None):
        if v_gradient is None:
            v_gradient = -0.006  # 6 degrees Celsius or Kelvins per 1 m
        return self.__interpolate_on_dem(self.aws.t_air, v_gradient)

    def __interpolate_pressure(self, v_gradient=None):
        if v_gradient is None:
            v_gradient = -0.1145  # Pa per 1 m
        return self.__interpolate_on_dem(self.aws.pressure, v_gradient)

    def __interpolate_e(self):
        """
        Interpolates partial pressure of water vapour (e)
        :return: np.array with distributed parameter
        """
        # delta_elev = self.dem - self.aws.elev
        return self.aws.e * 10 ** (-self.delta_dem / 6300)

    def __fill_array_with_one_value(self, value):
        """
        Assigns a given value to every non-nan cell of a grid
        :param value: any float
        :return: np.array
        """
        array = np.zeros_like(self.dem, dtype=np.float32)
        array[~np.isnan(self.dem)] = value
        array[np.isnan(self.dem)] = np.nan
        return array

    def __interpolate_on_dem(self, value, v_gradient):
        """
        Interpolates a single measurement of some meteorological parameter
        based on its vertical gradient and DEM
        :param value: measured value at the AWS
        :param v_gradient: vertical gradient
        :return: np.array with distributed parameter
        """
        return value + self.delta_dem * v_gradient

    def param_to_png(self, param_name):
        array = getattr(self, param_name)
        title = self.get_desc(param_name)
        units = self.get_units(param_name)
        show_me(array, title=title, units=units, show=False, verbose=True)


def to_kelvin(t_celsius):
    return t_celsius + 273.15


if __name__ == "__main__":
    test_aws_params = AwsParams(t_air=5, wind_speed=3, rel_humidity=0.80, pressure=1000, cloudiness=0.2, incoming_shortwave=250, z=1.6, elev=290, x=478342, y=8655635)
    print(test_aws_params)
    print(test_aws_params.Tz)
    print(test_aws_params.__dataclass_fields__['Tz'].metadata)
    # or:
    print(test_aws_params.get_units("Tz"))
    print(getattr(test_aws_params, "Tz"))
