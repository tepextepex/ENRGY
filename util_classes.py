from dataclasses import dataclass, field


@dataclass
class AwsParams:
    # meteo parameters:
    t_air: float = field(metadata={"units": "degree Celsius"})
    wind_speed: float = field(metadata={"units": "m per s"})
    rel_humidity: float = field(metadata={"units": "0-1"})
    pressure: float = field(metadata={"units": "hPa"})
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

    def __post_init__(self):
        t_surf = 0
        self.Tz = self.t_air + 273.15
        self.P = self.pressure * 100  # Pascals from hPa


if __name__ == "__main__":
    test_aws_params = AwsParams(t_air=5, wind_speed=3, rel_humidity=0.80, pressure=1000, cloudiness=0.2, incoming_shortwave=250, z=1.6, elev=290, x=478342, y=8655635)
    print(test_aws_params)
    print(test_aws_params.Tz)
    print(test_aws_params.__dataclass_fields__['Tz'].metadata)
