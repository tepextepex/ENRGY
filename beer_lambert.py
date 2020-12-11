"""
Computation of shortwave flux penetration into a glacier body according to Beer-Lambert law.

The penetration of shortwave radiation beneath the surface is taken into account
following Greuell and Oerlemans (1986) by assuming 36% (IR part of the solar spectrum)
is absorbed entirely at the surface.

Estimation of extinction coefficient for ice and snow follows
Bohren and Barkstrom (1974) and Greuell and Konzelmann (1994).
"""
import math

IR_IN_SOLAR_FLUX = 0.36  # infrared radiation is consumed by surface and does not penetrate under it


def absorbed_between(top_depth, bottom_depth, flux_in, density=900):
    """
    Computes a shortwave radiation flux which was absorbed inside
    a glacier body in a layer between top_depth and bottom_depth.
    :param top_depth: top boundary of a layer [m]
    :param bottom_depth: bottom boundary of a layer [m]
    :param flux_in: shortwave flux at a surface: incoming solar radiation multiplied by (1 - albedo) [W m-2]
    :param density: ice or snow density (0-1000) [kg m-3]
    :return: shortwave radiation flux absorbed inside the layer [W m-2]
    """
    absorbed = False
    top_out_flux = beer_lambert_for_glacier(flux_in, top_depth, density=density)
    # print(top_out_flux)
    bottom_out_flux = beer_lambert_for_glacier(flux_in, bottom_depth, density=density)
    # print(bottom_out_flux)
    absorbed = top_out_flux - bottom_out_flux if top_out_flux > bottom_out_flux else bottom_out_flux - top_out_flux
    if top_depth == 0 or bottom_depth == 0:
        absorbed += IR_IN_SOLAR_FLUX * flux_in
    return absorbed


def beer_lambert_for_glacier(flux_in, thickness, density=900):
    """
    Calculates shortwave radiation flux that penetrates under a layer with a given thickness.
    :param flux_in: shortwave flux at a top of glacier layer [W m-2]
    :param thickness: thickness of a glacier layer
    :param density: ice or snow density (0-1000) in the layer [kg m-3]
    :return: shortwave radiation flux at the bottom of the layer [W m-2]
    """
    flux_out = False
    try:
        if thickness > 0:
            k = __extinction_coef(density)
            flux_out = (1 - IR_IN_SOLAR_FLUX) * __beer_lambert(flux_in, k, thickness)
        elif thickness == 0:
            flux_out = (1 - IR_IN_SOLAR_FLUX) * flux_in
        else:
            raise ValueError
    except Exception as e:
        pass
    return flux_out


def thickness_for_ratio(ratio, extinction_coef, penetrated_ratio=None):
    """
    Computes material thickness where outgoing radiation flux becomes a [ratio] from incoming flux.
    :param ratio: a ratio of incoming radiation flux [0-1]
    :param extinction_coef: extinction coefficient for the given material and the given wave length [m-1]
    :param penetrated_ratio: a ratio of incoming flux which penetrates below surface
    :return: thickness (depth) where outgoing radiation flux becomes a [ratio] from incoming flux [m]
    """
    if penetrated_ratio is None:
        penetrated_ratio = 1
    thickness = False
    try:
        if (0 < ratio < 1) and (0 < penetrated_ratio <= 1):
            thickness = - math.log(ratio / penetrated_ratio) / extinction_coef
        elif (ratio == 1) and (0 < penetrated_ratio <= 1):
            thickness = 0
        else:
            raise ValueError
    except Exception as e:
        pass
    return thickness


def __beer_lambert(flux_in, extinction_coef, thickness):
    """
    Simple implementation of Beer-Lambert law.
    :param flux_in: incoming radiation flux [W m-2]
    :param extinction_coef: extinction coefficient for the given material and the given wave length [m-1]
    :param thickness: material thickness [m]
    :return: outgoing radiation flux [W m-2]
    """
    flux_out = False
    try:
        flux_out = flux_in * math.exp(-extinction_coef * thickness)
    except Exception as e:
        pass
    return flux_out


def __extinction_coef(ice_density):
    """
    Computes an extinction coefficient for shortwave radiation inside water ice/snow.
    :param ice_density: ice or snow density [kg m-3], valid range from 0 to 1000
    :return: extinction coefficient for shortwave radiation flux [m-1]
    """
    k = False
    try:
        k = 20 if ice_density <= 450 else -7 / 180 * ice_density + 37.5
        if ice_density > 1000:
            raise ValueError
    except Exception as e:
        pass
    return k


if __name__ == "__main__":
    in_flux = 100  # incoming solar radiation multiplied by (1 - albedo) [W/m-2]
    a = [x * 0.1 for x in range(0, 10)]
    for x in range(0, 10):
        print("Depth: %.2f-%.2f" % (x * 0.2, x * 0.2 + 0.2))
        absorbed = absorbed_between(x * 0.2, x * 0.2 + 0.2, 100)
        print("Percent of flux absorbed inside the layer: %.1f" % absorbed)
