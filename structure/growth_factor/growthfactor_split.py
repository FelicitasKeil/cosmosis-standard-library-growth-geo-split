from cosmosis.datablock import names, option_section as opt
import numpy as np
from scipy import differentiate, integrate, interpolate

cosmo = names.cosmological_parameters


def setup(options):
    section = opt
    verbose = options.get_bool(section, "verbose", default=False)
    wmodel = options.get_int(section, "w_model", default=0)
    zmin = options.get_double(section, "zmin", default=0.0)
    zmax = options.get_double(section, "zmax", default=4.0)
    dz = options.get_double(section, "dz", default=0.01)
    return zmin, zmax, dz, verbose, wmodel


def deriv(lna, y, h0, om, om_geo, ok, w0, wa):
    deriv = np.zeros([len(y)])
    z = np.exp(-lna) - 1
    h = hubble(lna, h0, om_geo, om_geo, ok, w0, wa)
    h_prime = hPrimelna(lna, h0, om_geo, om_geo, ok, w0, wa)
    c1 = 4 + h_prime / h
    c2 = (3 + h_prime / h - 3 / 2 * om * (1 + z) ** 3 / e_z(lna, h0, om, om_geo,
                                                            ok, w0, wa
                                                            ) ** 2)
    deriv[0] = y[1]
    deriv[1] = - c1 * y[1] - c2 * y[0]
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

    h0 = block[cosmo, "h0"]
    om_geo = block[cosmo, "omega_m"]
    om_growth = block[cosmo, "omega_m_growth"]
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
        print(
            "om_geo, om_growth, ok, h0, w0, wa",
            om_geo, om_growth, ok, h0, w0, wa)
        
    
    lna_vec_ODE = np.linspace(-3, 0, 1000)

    sol_geo = solve_growth_ODE(h0, om_geo, om_geo, ok, w0, wa, lna_vec_ODE)
    sol_growth = solve_growth_ODE(h0, om_growth, om_geo, ok, w0,
                                              wa, lna_vec_ODE)
    
    g_lna_geo_interp = interpolate.interp1d(np.exp(-sol_geo.t)-1, sol_geo.y[0])
    g_z_geo = g_lna_geo_interp(z)
    g_prime_lna_geo_interp = interpolate.interp1d(np.exp(-sol_geo.t)-1, sol_geo.y[1])
    g_prime_z_geo = g_prime_lna_geo_interp(z)
    f_z_geo = 1 + g_prime_z_geo / g_z_geo

    g_lna_growth_interp = interpolate.interp1d(np.exp(-sol_geo.t)-1, sol_growth.y[0])
    g_z_growth = g_lna_growth_interp(z)
    g_prime_lna_growth_interp = interpolate.interp1d(np.exp(-sol_geo.t)-1,
                                                     sol_growth.y[1])
    g_prime_z_growth = g_prime_lna_growth_interp(z)
    f_z_growth = 1 + g_prime_z_growth / g_z_growth

    block[names.growth_parameters, "G_z_geo"] = g_z_geo
    block[names.growth_parameters, "G_z_growth"] = g_z_growth
    block[names.growth_parameters, "f_z_geo"] = f_z_geo
    block[names.growth_parameters, "f_z_growth"] = f_z_growth

    k_h = block[names.matter_power_nl, "k_h"]
    # non-linear boost factor
    boost_factor = (
        block[names.matter_power_nl, "P_K"] / block[names.matter_power_lin, "P_K"]
    )

    # Rescale power spectrum with the growth at every redshift
    P_k_temp = np.zeros((len(z), len(k_h)))

    for i in range(len(z)):
        rescale_fac = g_z_growth[i] ** 2 / g_z_geo[i] ** 2
        P_k_temp[i, :] = rescale_fac * block[names.matter_power_lin, "P_K"][i, :]

    # rescale sigma 8 (previously from the EBS) with the growth
    block[names.growth_parameters, "sigma_8"] = (
        g_z_growth / g_z_geo * block[names.growth_parameters, "sigma_8"]
    )
    block[names.growth_parameters, "fsigma_8_growth"] = (
        block[names.growth_parameters, "f_z_growth"]
        * block[names.growth_parameters, "sigma_8"]
    )

    block[names.matter_power_lin, "P_K"] = P_k_temp
    # Multiplying by the boost matrix to get the non-linear power spectrum
    block[names.matter_power_nl, "P_K"] = (
        boost_factor * block[names.matter_power_lin, "P_K"]
    )
    block[names.matter_power_nl, "B"] = boost_factor

    return 0
