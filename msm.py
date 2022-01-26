from parameter_classes import CONST
import numpy as np
import matplotlib.pyplot as plt


def calc_gradients(depths, temps):
    g = []
    for t0, t1, d in zip(temps, temps[1:], depths):
        # print(t0, t1, d)
        grad = (t1 - t0) / d  # temperature gradient downwards, K / m
        # print("Gradient: ", grad)
        g.append(grad)
    return g


def tick(depths, temps, timestep, flux=None, snow_depth=None):
    """

    :param depths: these are layer thicknesses actually, not the depths-below-surface [m]
    :param temps: temperatures of boundaries between layers [K] or [degree Celsius]
    :param timestep: time step of modelling [s]
    :param flux: atmospheric heat flux, applied to the uppermost layer [W/m^2]
    :param snow_depth: snow layer depth relative to the topmost boundary of temps[0] [m, not m w.e.!!]
    :return: updated temperatures
    """

    c = CONST["specific_heat_capacity_ice"]
    k_ice = CONST["thermal_diffusivity_ice"]
    k_snow = CONST["thermal_diffusivity_snow"]
    rho_ice = CONST["ice_density"]
    rho_snow = CONST["snow_density"]

    if flux is None:
        flux = 0

    new_temps = []
    g = calc_gradients(depths, temps)

    for i in range(0, len(temps) - 1):
        # first we should define bulk density and bulk diffusivity of each layer based on snow/ice ratio:
        if snow_depth is None:
            k = k_ice
            rho = rho_ice
        else:
            if snow_depth >= depths[i]:
                snow_ratio = 1
            else:
                snow_ratio = snow_depth / depths[i]
            k = snow_ratio * k_snow + (1 - snow_ratio) * k_ice
            rho = snow_ratio * rho_snow + (1 - snow_ratio) * rho_ice
            snow_depth -= depths[i]
            snow_depth = 0 if snow_depth < 0 else snow_depth
        ################################################################################################
        if i == 0:
            delta_t = k * g[i] / depths[i]
            ground_flux = k * g[i] * c * rho
            # print(ground_flux, "W/m^2")

            full_flux = flux + ground_flux
            # print(full_flux, "W/m^2")
            q0 = -temps[i] * c * rho * depths[i] / timestep
            # print(q0, "W/m^2")

            # if the full heat flux is not enough to heat the layer up to the melting temperature,
            # then the heat available for melt is zero:
            qm = 0 if full_flux <= q0 else full_flux - q0
            # print("Qm", qm, "W/m^2")

            delta_t += (flux - qm) / (c * rho * depths[i])
        else:
            delta_t = k * (g[i] - g[i - 1]) / depths[i]
            # print("delta t is", delta_t)
        new_temps.append(temps[i] + delta_t * timestep)
    new_temps.append(temps[-1])  # temperature of the deepest layer is constant
    return new_temps, qm


def update_layers(depths, temps, surf_lowering):
    """

    :param depths: the old layer thicknesses [m]
    :param temps: the old temperatures of tha layer boundaries [K] or [degree Celsius]
    :param surf_lowering: bulk surface lowering (snow + ice) in meters
    :return:
    """
    if surf_lowering <= 0:
        return depths, temps
    else:
        for i in range(len(depths)):
            if depths[i] > surf_lowering:
                depths[i] -= surf_lowering
                temps[i] = 0
                break
            else:
                surf_lowering -= depths[i]
                depths[i] = 0
                temps[i] = np.nan
    return depths, temps


if __name__ == "__main__":
    depths = [0.22, 0.78, 4.5]
    temps = [-2.0, -2.2, -3.0, -6.9]
    temps = [x - 2 for x in temps]

    t_list = [temps[:]]
    d_list = [depths[:]]

    x = np.arange(24 * 14)
    atmo_forcing = np.sin(2 * np.pi * 24 * x / 580) * 30 + 25
    time_step = 3600  # 1 hour

    snow_depth = 0.10  # meters, not m w.e.(!)
    snow_depths = [snow_depth]

    ice_depth = 0.00
    ice_depths = [ice_depth]

    for i, atmo_flux in zip(x, atmo_forcing):

        ####################################
        ### TEMPERATURE CHANGING ROUTINE  ##
        ####################################
        temps, melt_flux = tick(depths, temps, time_step, flux=atmo_flux, snow_depth=snow_depth)
        temps = [round(x, 3) for x in temps]
        t_list.append(temps[:])
        d_list.append(depths[:])
        # print("result", temps)
        ####################################
        latent_heat_of_fusion = CONST["latent_heat_of_fusion"]
        ice_density = CONST["ice_density"]
        snow_density = CONST["snow_density"]

        ####################################
        ###     SURFACE MELT ROUTINE     ###
        ####################################
        melt_rate_kg = melt_flux / latent_heat_of_fusion  # melt amount per second in kg
        melt_rate_kg = 0 if melt_rate_kg < 0 else melt_rate_kg  # since negative melt is not possible
        melt_amount_we = time_step * melt_rate_kg / 1000  # melt amount in m w.e. (1 m w.e. -> 1000 kg m-2)
        print()

        swe = snow_depth * snow_density / 1000
        print("Actual SWE:", swe, "m w.e.")
        if melt_amount_we > swe:
            snow_melt_amount_we = swe
            # and the ice melt begins:
            ice_melt_amount_we = melt_amount_we - snow_melt_amount_we  # total melt amount in w.e. minus snow melt amount
        else:
            snow_melt_amount_we = melt_amount_we
            # and the ice was preserved from melt:
            ice_melt_amount_we = 0
        swe -= snow_melt_amount_we  # updates snow water equivalent available for melt, can't be below zero
        print("Snow melt:", snow_melt_amount_we, "m w.e.")
        print("Ice melt:", ice_melt_amount_we, "m w.e.")

        ####################################
        ###   SURFACE LOWERING ROUTINE   ###
        ####################################
        surf_lowering = snow_depth - swe / snow_density * 1000  # the old snow depth minus the new
        print("Snow surface lowering:", surf_lowering, "m")
        snow_depth = swe / snow_density * 1000  # updates snow thickness available for melt
        snow_depths.append(snow_depth)
        print("New snow depth:", snow_depth, "m")
        surf_lowering += ice_melt_amount_we / ice_density * 1000
        print("Total surface lowering:", surf_lowering, "m")
        ice_depths.append(ice_depths[-1] - ice_melt_amount_we / ice_density * 1000)

        ####################################
        ###   DEPTHS AND TEMPS UPDATING  ###
        ####################################
        depths, temps = update_layers(depths, temps, surf_lowering)
        ###################################

    fig, ax = plt.subplots(4, 1)
    t_list = np.transpose(t_list)
    for t in t_list:
        ax[0].plot(t)
    d_list = np.transpose(d_list)
    for d in d_list:
        ax[1].plot(d)

    ax[2].plot(atmo_forcing)  # plots the sine wave for atmo forcing
    ax[3].plot(snow_depths, c="lightblue", label="snow")
    ax[3].plot(ice_depths, c="darkblue", label="ice")
    plt.show()
