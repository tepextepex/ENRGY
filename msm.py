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


def tick(depths, temps, timestep, flux=None):
    """

    :param depths: these are layer thicknesses actually, not the depths-below-surface [m]
    :param temps: temperatures of boundaries between layers [K] or [degree Celsuis]
    :param timestep: time step of modelling [s]
    :param flux: atmospheric heat flux, applied to the uppermost layer [W/m^2]
    :return: updated temperatures
    """

    c = CONST["specific_heat_capacity_ice"]
    k = CONST["thermal_diffusivity_ice"]
    rho = CONST["ice_density"]

    new_temps = []
    g = calc_gradients(depths, temps)

    for i in range(0, len(temps) - 1):
        if i == 0:
            delta_t = k * g[i] / depths[i]
            ground_flux = k * g[i] * c * rho
            # print(ground_flux, "W/m^2")
            if flux is not None:

                full_flux = flux + ground_flux
                # print(full_flux, "W/m^2")
                q0 = -temps[i] * c * rho * depths[i] / timestep
                # print(q0, "W/m^2")
                # if the full heat flux is not enough to heat the layer up to the melting temperature,
                # then the heat available for melt is zero:
                qm = 0 if full_flux <= q0 else full_flux - q0
                print("Qm", qm, "W/m^2")

                delta_t += (flux - qm) / (c * rho * depths[i])
        else:
            delta_t = k * (g[i] - g[i - 1]) / depths[i]
            # print("delta t is", delta_t)
        new_temps.append(temps[i] + delta_t * timestep)
    new_temps.append(temps[-1])  # temperature of the deepest layer is constant
    return new_temps


if __name__ == "__main__":
    depths = [0.22, 0.78, 4.5]
    temps = [-2.0, -2.2, -3.0, -6.9]

    t_list = []
    t_list.append(temps)

    x = np.arange(24 * 30)
    atmo_forcing = np.sin(2 * np.pi * 24 * x / 580) * 30 + 40

    for i, atmo_flux in zip(x, atmo_forcing):
        temps = tick(depths, temps, 3600, flux=atmo_flux)
        temps = [round(x, 3) for x in temps]
        t_list.append(temps)
        # print("result", temps)
    t_list = np.transpose(t_list)
    fig, ax = plt.subplots(2, 1)
    for t in t_list:
        ax[0].plot(t)
    ax[1].plot(atmo_forcing)  # plots the sine wave for atmo forcing

    plt.show()

