from parameter_classes import CONST
import numpy as np
import matplotlib.pyplot as plt

latent_heat_of_fusion = CONST["latent_heat_of_fusion"]
ice_density = CONST["ice_density"]
snow_density = CONST["snow_density"]


def my_print(flux_title, flux_value):
    if isinstance(flux_value, np.ndarray):
        pass
    else:
        print("%s:" % flux_title, round(flux_value, 1), "W/m^2")


def calc_gradients(depths, temps):
    g = []
    for t0, t1, d in zip(temps, temps[1:], depths):
        # print(t0, t1, d)
        if d == 0:
            grad = np.nan
        else:
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

    surf = True  # used to find the surface layer. Once found, changed to False
    qm = np.nan
    for i in range(0, len(temps) - 1):
        # first we should define bulk density and bulk diffusivity of each layer based on snow/ice ratio:
        if snow_depth is None:
            k = k_ice
            rho = rho_ice
        else:
            if isinstance(snow_depth, np.ndarray):
                snow_ratio = np.where(snow_depth > depths[i], 1, snow_depth / depths[i])
            else:
                if snow_depth >= depths[i]:
                    snow_ratio = 1
                else:
                    snow_ratio = snow_depth / depths[i]
            k = snow_ratio * k_snow + (1 - snow_ratio) * k_ice
            rho = snow_ratio * rho_snow + (1 - snow_ratio) * rho_ice
            snow_depth -= depths[i]
            if isinstance(snow_depth, np.ndarray):
                snow_depth[snow_depth < 0] = 0
            else:
                snow_depth = 0 if snow_depth < 0 else snow_depth
        ################################################################################################
        if depths[i] == 0:
            new_temps.append(temps[i])
            continue  # all the zero-thickness layers do not exist any more, just skip them
        if surf:  # executed for the first layer with a non-zero thickness
            # delta_t = k * g[i] / depths[i]
            ground_flux = k * g[i] * c * rho
            print()
            my_print("Ground flux", ground_flux)

            full_flux = flux + ground_flux
            my_print("Full flux (atmo + ground)", full_flux)
            q0 = -temps[i] * c * rho * depths[i] / timestep
            my_print("Heat needed to increase the temperature up to melting point", q0)

            # computing heat flux available for melt (qm), can't be below zero:
            qm = full_flux - q0
            if isinstance(qm, np.ndarray):
                qm[qm < 0] = 0
            else:
                qm = 0 if qm < 0 else qm
            my_print("Heat available for melt", qm)
            delta_t = (full_flux - qm) / (c * rho * depths[i])
            my_print("Flux which changes the layer temperature", (full_flux - qm))

            surf = False
        else:  # other layers below the surface
            delta_t = k * (g[i] - g[i - 1]) / depths[i]
            # print("delta t is", delta_t)
        new_temps.append(temps[i] + delta_t * timestep)
    new_temps.append(temps[-1])  # temperature of the deepest layer is constant
    return new_temps, qm, ground_flux


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
                # print(len(depths), temps)
                temps[i] = np.nan
    # print(depths, temps)
    thickness_threshold = 0.03  # too thin layers produce HUGE gradients inappropriate for our time step
    depths, temps = filter_layers(depths, temps, thickness_threshold)
    return depths, temps


def filter_layers(depths, temps, threshold):
    """

    :param depths:
    :param temps:
    :param threshold:
    :return:
    """
    for i in range(len(depths) - 1):  # the deepest layer remains unchanged, aware to set it deep enough
        if depths[i] == 0:
            continue
        if depths[i] < threshold:
            depths[i + 1] = depths[i + 1] + depths[i]
            depths[i] = 0
            temps[i + 1] = temps[i]
            temps[i] = np.nan
            print("Filtered", depths, temps)
            break  # or should we check each layer, not only the first?
    return depths, temps


def calc_melt_old(melt_flux, swe, time_step):
    """
    Assuming that a glacier surface consist of two layers - 1) the snow with a given SWE on top of the 2) glacier ice,
    computes the melt amount of these two layers separately. First the snow is being melt and after the SWE becomes zero,
    the ice ablation starts.
    :param time_step:
    :param melt_flux: heat flux available for melt [W m-2]
    :param swe: array of the snow water equivalent [m w.e.]
    :return:
    """
    melt_rate_kg = melt_flux / latent_heat_of_fusion  # melt amount per second in kg of water [kg s-1]

    # for the distributed model:
    if isinstance(melt_rate_kg, np.ndarray):
        melt_rate_kg[melt_rate_kg < 0] = 0
        melt_amount_we = time_step * melt_rate_kg / 1000  # melt amount in [m w.e.] (1 m w.e. -> 1000 kg m-2)
        snow_melt_amount_we = np.where(melt_amount_we > swe, swe, melt_amount_we)
        ice_melt_amount_we = np.where(melt_amount_we > swe, melt_amount_we - snow_melt_amount_we, 0)

    # melt_rate_kg may be float for lumped modelling:
    else:
        melt_rate_kg = 0 if melt_rate_kg < 0 else melt_rate_kg  # since negative melt is not possible
        melt_amount_we = time_step * melt_rate_kg / 1000  # melt amount in [m w.e.] (1 m w.e. -> 1000 kg m-2)
        if melt_amount_we > swe:
            snow_melt_amount_we = swe  # snow melt can't exceed the real swe
            # after all the snow is melt down, the ice ablation begins:
            ice_melt_amount_we = melt_amount_we - snow_melt_amount_we  # total melt amount in w.e. minus snow melt amount
        else:
            snow_melt_amount_we = melt_amount_we
            # and the ice was preserved from melt:
            ice_melt_amount_we = 0

    return snow_melt_amount_we, ice_melt_amount_we


def calc_melt(melt_flux, swe, time_step):
    q = melt_flux * time_step  # the amount of heat in J kg-1
    # print("Q is", q)
    total_melt_kg = q / latent_heat_of_fusion  # kg m-2
    total_melt_we = total_melt_kg / 1000  # meters of w.e.
    if isinstance(total_melt_we, np.ndarray):
        snow_melt_we = np.where(total_melt_we > swe, swe, total_melt_we)  # snow melt could not exceed the current snow remnant
    else:
        snow_melt_we = swe if total_melt_we > swe else total_melt_we
    ice_melt_we = total_melt_we - snow_melt_we
    return snow_melt_we, ice_melt_we


def report(t_list, d_list, atmo_forcing):
    fig, ax = plt.subplots(4, 1)
    """
    for t in t_list:
        print(t)
    """
    t_list = np.transpose(t_list)
    for t in t_list:
        ax[0].plot(t)
    ax[0].title.set_text("Temperature by layers")

    d_list = np.transpose(d_list)
    # print(len(x))

    label = 1
    for d in d_list[:-1]:
        ax[1].plot(d, label="Layer %s" % label)
        label += 1

    ax[1].legend()
    ax[1].title.set_text("Layer thickness")

    ax[2].plot(atmo_forcing, label="atmo heat flux")  # plots the sine wave for atmo forcing
    ax[2].legend()
    ax[2].title.set_text("Atmospheric heat flux")
    ax[3].plot(snow_depths, c="lightblue", label="snow")
    ax[3].plot(ice_depths, c="darkblue", label="ice")
    ax[3].legend()
    ax[3].title.set_text("Surface lowering (m)")

    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # depths = [0.5, 0.5, 0.5, 0.5, 3.5]
    # temps = [-9.81, -5.5, -6.0, -6.75, -6.62, -4.68]

    depths = [0.10, 0.4, 0.5, 0.5, 0.5, 3.0]
    temps = [-9.81, -8.95, -5.5, -6.75, -6.62, -6.18, -4.68]  # 24 April 00:00

    t_list = [temps[:]]  # for plotting
    d_list = [depths[:]]  # for plotting

    days = 45
    x = np.arange(24 * days)
    #atmo_forcing = np.sin((x % 24) * np.pi / 12) * 100 + 50
    atmo_forcing = np.sin((x % 24) * np.pi / 12) * 100 + 20
    time_step = 3600 # 1 hour

    snow_depth = 1.00  # meters, not m w.e.(!)
    snow_depths = [snow_depth]  # for plotting

    ice_depth = 0.00
    ice_depths = [ice_depth]  # for plotting

    for atmo_flux in atmo_forcing:
        ####################################
        ### TEMPERATURE CHANGING ROUTINE  ##
        ####################################
        temps, melt_flux, glacier_flux = tick(depths, temps, time_step, flux=atmo_flux, snow_depth=snow_depth)
        temps = [round(x, 3) for x in temps]
        # print(temps, melt_flux, glacier_flux)
        t_list.append(temps[:])
        d_list.append(depths[:])

        ####################################
        ###     SURFACE MELT ROUTINE     ###
        ####################################
        swe = snow_depth * snow_density / 1000
        snow_melt_amount_we, ice_melt_amount_we = calc_melt(melt_flux, swe, time_step)
        swe -= snow_melt_amount_we  # updates snow water equivalent available for melt, can't be below zero
        print("Snow melt:", snow_melt_amount_we, "m w.e.")
        print("Ice melt:", ice_melt_amount_we, "m w.e.")

        ####################################
        ###   SURFACE LOWERING ROUTINE   ###
        ####################################
        prev_depth = snow_depth
        snow_depth = swe / snow_density * 1000  # updates snow thickness available for melt
        surf_lowering = prev_depth - snow_depth  # the old snow depth minus the new
        print("Snow surface lowering:", surf_lowering, "m")
        print("New snow depth:", snow_depth, "m")
        ice_lowering = ice_melt_amount_we / ice_density * 1000  # a portion of the surface lowering due to ice melt
        surf_lowering += ice_lowering
        print("Snow+ice surface lowering:", surf_lowering, "m")

        # for plotting:
        snow_depths.append(snow_depth)
        ice_depths.append(ice_depths[-1] - ice_lowering)

        ####################################
        ###   DEPTHS AND TEMPS UPDATING  ###
        ####################################
        # depths, temps = update_layers(depths, temps, surf_lowering)
        ###################################

    report(t_list, d_list, atmo_forcing)  # plots the final result
