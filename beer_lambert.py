"""
Computation of shortwave flux penetration into a glacier body according to Beer-Lambert law.

The penetration of shortwave radiation beneath the surface is taken into account
following Greuell and Oerlemans (1986) by assuming 36% (IR part of the solar spectrum)
is absorbed entirely at the surface.

Computation of extinction coefficient for ice and snow follows
Bohren and Barkstrom (1974) and Greuell and Konzelmann (1994).
"""
import math
import matplotlib.pyplot as plt

IR_IN_SOLAR_FLUX = 0.36  # infrared radiation is consumed by surface and does not penetrate under it


def absorbed_between(top_depth, bottom_depth, flux_in, density=900):
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

    :param flux_in:
    :param thickness:
    :param density:
    :return:
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
    Computes material thickness where outgoing radiation flux becomes a [ratio] from incoming flux
    :return: thickness (depth) [m]
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

    :param flux_in:
    :param extinction_coef:
    :param thickness:
    :return:
    """
    flux_out = False
    try:
        flux_out = flux_in * math.exp(-extinction_coef * thickness)
    except Exception as e:
        pass
    return flux_out


def __extinction_coef(ice_density):
    """

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
    # print(__beer_lambert(100, 2.5, 0.25))
    # print(__extinction_coef(850))
    # print(beer_lambert_for_glacier(100, 0.22, 900))
    in_flux = 100
    """
    depths = [-0.05 * x for x in range(0, 45)]
    density = 880
    fluxes = [beer_lambert_for_glacier(in_flux, -x, density=density) for x in depths]
    k = __extinction_coef(density)
    one_percent_depth = thickness_for_ratio(0.01, k, penetrated_ratio=0.64)
    print("One percent flux depth is %.2f m" % one_percent_depth)
    # print(depths)
    # print(fluxes)

    plt.style.use("seaborn")
    plt.figure(figsize=(4, 6))
    plt.plot(fluxes, depths, label="Shortwave flux, $\\rho$=%s $kg m^{-2}$" % density)
    plt.text(0, 0, "1%% of incoming flux is at %.2f m" % one_percent_depth, fontsize=9)
    plt.legend()
    plt.xlim(-1, 100)
    plt.xlabel("Percent of incoming radiation flux, %")
    plt.ylabel("Depth below glacier surface, m")
    plt.tight_layout()
    plt.show()
    """
    a = [x * 0.1 for x in range(0, 10)]
    for x in range(0, 10):
        print("Depth: %.2f-%.2f" % (x * 0.2, x * 0.2 + 0.2))
        absorbed = absorbed_between(x * 0.2, x * 0.2 + 0.2, 100)
        print("Percent of flux absorbed inside the layer: %.1f" % absorbed)
