from cosmosis.datablock import names, option_section as opt
from cosmosis.datablock.cosmosis_py import errors
import numpy as np
import matplotlib.pyplot as plt
from scipy import differentiate
from scipy import integrate
from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section
import pickle

cosmo = section_names.cosmological_parameters


def setup(options):
    section = option_section
    verbose = options.get_bool(section, "verbose", default=False)
    wmodel = options.get_int(section, "w_model", default=0)
    zmin = options.get_double(section, "zmin", default=0.0)
    zmax = options.get_double(section, "zmax", default=4.0)
    dz = options.get_double(section, "dz", default=0.01)
    return zmin, zmax, dz, verbose, wmodel


def deriv(lna, y, h0, om, om_geo, ok, w0, wa):
    deriv = np.zeros([len(y)])
    z = np.exp(-lna) - 1
    deriv[0] = y[1]
    c1 = ( 4 + hPrimelna(lna, h0, om_geo, om_geo, ok, w0, wa)
            / hubble(lna, h0, om_geo, om_geo, ok, w0, wa))
    c2 = ( 3 + hPrimelna(lna, h0, om_geo, om_geo, ok, w0, wa)
            / hubble(lna, h0, om_geo, om_geo, ok, w0, wa)
            - 3 / 2 * om * (1 + z) ** 3
            * e_z(lna, h0, om, om_geo, ok, w0, wa) ** (-2))
    deriv[1] = (- c1 * y[1] - c2 * y[0])
    return deriv


def w_int(lna, h0, om, om_geo, ok, w0, wa):
    a = np.exp(lna)
    # analytic integral for the w0wa model, see e.g. Equ.(9) of 1910.09273
    return 3.0 * (wa * (a - 1.0) - lna * (1 + w0 + wa))


def e_z(lna, h0, om, om_geo, ok, w0, wa):
    z = np.exp(-lna) - 1
    return np.sqrt(
        om * (1 + z) ** 3 + ok * (1 + z
                                  ) ** 2 + (1 - om - ok
                                            ) * np.exp(w_int(lna, h0, om,
                                                             om_geo, ok, w0,
                                                             wa)))


def hubble(lna, h0, om, om_geo, ok, w0, wa):
    return 100.0 * h0 * e_z(lna, h0, om_geo, om_geo, ok, w0, wa)


def hPrimelna(lna, h0, om, om_geo, ok, w0, wa):
    hPrime = differentiate.derivative(hubble, lna,
                                      args=(h0, om_geo, om_geo, ok, w0, wa))
    return hPrime.df


def solve_growth_ODE(h0, om, om_geo, ok, w0, wa, lna_vec):
    zinit = 1000
    w_int_init = w_int(np.log(1 / (1 + zinit)), h0, om, om_geo, ok, w0, wa)
    e_z_init = e_z(np.log(1 / (1 + zinit)), h0, om, om_geo, ok, w0, wa)
    # initial values for the ODE from Miranda et al. 1712.04289, page 4
    y01 = (-6 / 5 * (1 - om - ok) * np.exp(w_int_init) * e_z_init ** (-2))
    y0 = np.array([1.0, y01])

    sol = integrate.solve_ivp(
        deriv,
        (np.min(lna_vec), np.max(lna_vec)),
        y0,
        # method="DOP853",
        t_eval=lna_vec,
        args=(h0, om, om_geo, ok, w0, wa))

    return sol


# test function to compute G_growth^2/G_geo^2
def f_gamma_times_a(z, h0, om, om_geo, ok, w0, wa):
    return (om*(1+z) ** 3 / e_z(np.log(1/(1+z)), h0, om, om_geo, ok, w0, wa
                                ) ** 2)**0.55/(1+z)


def execute(block, config):
    zmin, zmax, dz, verbose, wmodel = config
    
    z = block[names.matter_power_nl, "z"]

    lna_vec = np.log(1/(1+z))[::-1]

    h0 = block[cosmo, "h0"]
    om_geo = block[cosmo, "omega_m"]
    om_growth = block[cosmo, "omega_m_growth"]
    om_early = block[cosmo, "omega_m_early"]
    ok = block[cosmo, "omega_k"]
    if wmodel == 0:
        w0 = block[cosmo, "w"]
        wa = block[cosmo, "wa"]
    else:
        raise ValueError(
            "You dark energy model is not supported. For now only"
            "w0wa is supported by the growth-geometry split module."
        )
    if verbose:
        print("om_geo, om_growth, om_early, ok, h0, w0, wa", om_geo,
              om_growth, om_early, ok, h0, w0, wa)

    g_early = solve_growth_ODE(h0, om_early, om_geo, ok, w0, wa, lna_vec)
    g_growth = solve_growth_ODE(h0, om_growth, om_geo, ok, w0, wa, lna_vec)
    # reverse the arrays to store them as a function of z
    g_z_early = g_early.y[0][::-1]
    g_z_growth = g_growth.y[0][::-1]

    block[names.growth_parameters, "G_z_early"] = g_z_early
    block[names.growth_parameters, "G_z_growth"] = g_z_growth
    block[names.growth_parameters,
          "f_z_early"] = 1 + g_early.y[1][::-1] / g_early.y[0][::-1]
    block[names.growth_parameters,
          "f_z_growth"] = 1 + g_growth.y[1][::-1] / g_growth.y[0][::-1]
    # rescale sigma 8 (previously from the EBS) with the growth
    block[names.growth_parameters,
          "sigma_8"] = g_z_growth / g_z_early * block[names.growth_parameters,
                                                    "sigma_8"]
    block[names.growth_parameters,
          "fsigma_8_growth"] = block[names.growth_parameters,
                                     "f_z_growth"] * block[names.growth_parameters,
                                                           "sigma_8"]
    
    k_h=block[names.matter_power_nl, "k_h"]
    # non-linear boost factor
    boost_factor = block[names.matter_power_nl,
                         "P_K"]/block[names.matter_power_lin, "P_K"]

    # Rescale power spectrum with the growth at every redshift
    P_k_temp = np.zeros((len(z), len(k_h)))

    for i in range(len(z)):
        rescale_fac = g_z_growth[i]**2/g_z_early[i]**2
        P_k_temp[i, :] = rescale_fac * block[names.matter_power_lin,
                                             "P_K"][i, :]

    block[names.matter_power_lin, "P_K"] = P_k_temp
    # Multiplying by the boost matrix to get the non-linear power spectrum
    block[names.matter_power_nl,
          "P_K"] = boost_factor * block[names.matter_power_lin, "P_K"]
    block[names.matter_power_nl,"B"] = boost_factor

    # pickle the rescaled power spectrum
    # with open("mps_om-geo-025-om-growth-03.pkl", "wb") as out_pkl:
    #     pickle.dump(block[names.matter_power_nl, "P_K"], out_pkl)

    with open("mps_om-geo-03-om-growth-03.pkl", "rb") as in_pkl:
        mps_geo_03_gr_03 = pickle.load(in_pkl)

    with open("mps_om-geo-03-om-growth-025.pkl", "rb") as in_pkl:
        mps_geo_03_gr_025 = pickle.load(in_pkl)

    with open("mps_om-geo-03-om-growth-035.pkl", "rb") as in_pkl:
        mps_geo_03_gr_035 = pickle.load(in_pkl)

    with open("mps_om-geo-025-om-growth-03.pkl", "rb") as in_pkl:
        mps_geo_025_gr_03 = pickle.load(in_pkl)

    with open("mps_om-geo-035-om-growth-03.pkl", "rb") as in_pkl:
        mps_geo_035_gr_03 = pickle.load(in_pkl)

    plt.hlines(
        1, np.min(k_h), np.max(k_h), label="$\\Omega_{\\rm m}=0.3$", color="black"
    )
    plt.plot(
        k_h,
        mps_geo_03_gr_025[0, :] / mps_geo_03_gr_03[0, :],
        label="$\\Omega^{early}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.25$",
    )
    plt.plot(
        k_h,
        mps_geo_03_gr_035[0, :] / mps_geo_03_gr_03[0, :],
        label="$\\Omega^{early}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.35$",
    )
    plt.plot(
        k_h,
        mps_geo_025_gr_03[0, :] / mps_geo_03_gr_03[0, :],
        label="$\\Omega^{early}_{\\rm m}=0.25, \\Omega^{growth}_{\\rm m}=0.3$",
    )
    plt.plot(
        k_h,
        mps_geo_035_gr_03[0, :] / mps_geo_03_gr_03[0, :],
        label="$\\Omega^{early}_{\\rm m}=0.35, \\Omega^{growth}_{\\rm m}=0.3$",
    )
    plt.title("Non-linear Power Spectrum Ratio at z=0")
    plt.xlabel("k [h/Mpc]")
    plt.ylabel("$P_{\\rm rescaled}(k)/P(k)$")
    plt.xscale("log")
    plt.ylim(0.25, 1.6)
    plt.legend()
    plt.savefig("power_spectra_ratio_k_early.png", dpi=300)
    plt.clf()

    # plt.rcParams.update({"font.size": 13})
    plt.plot(
        k_h,
        mps_geo_03_gr_03[0, :],
        label="$\\Omega_{\\rm m}=0.3$",
        color="black",
        linewidth=0.9,
    )
    plt.plot(
        k_h,
        mps_geo_03_gr_025[0, :],
        label="$\\Omega^{early}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.25$",
        linewidth=0.9,
    )
    plt.plot(
        k_h,
        mps_geo_03_gr_035[0, :],
        label="$\\Omega^{early}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.35$",
        linewidth=0.9,
    )
    plt.plot(
        k_h,
        mps_geo_025_gr_03[0, :],
        label="$\\Omega^{early}_{\\rm m}=0.25, \\Omega^{growth}_{\\rm m}=0.3$",
        linewidth=0.9,
    )
    plt.plot(
        k_h,
        mps_geo_035_gr_03[0, :],
        label="$\\Omega^{early}_{\\rm m}=0.35, \\Omega^{growth}_{\\rm m}=0.3$",
        linewidth=0.9,
    )
    plt.title("Non-linear Power Spectrum at z=0")
    plt.xlabel("k [h/Mpc]")
    plt.ylabel("$P(k) [(h/Mpc)^3]$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    # plt.tight_layout()
    plt.savefig("compare_power_spectra_early.png", dpi=300)
    plt.clf()

    return 0
