from cosmosis.gaussian_likelihood import GaussianLikelihood
from cosmosis.datablock import names, option_section as opt
import numpy as np
import scipy.interpolate
import camb

# Data from Table 1 of https://arxiv.org/pdf/2302.12911v2
EARLY_PLANCK_DATA = {
    "Omega_b_h2": {
        "kind": "omega_b",
        "mean": 0.02223,
        "error": 0.00015,
    },
    "Omega_c_h2": {
        "kind": "omega_c",
        "mean": 0.1192,
        "error": 0.0013,
    },
    "A_s": {
        "kind": "1e9_A_s_exp_-2tau",
        "mean": 1.873,
        "error": 0.012,
    },
    "n_s": {
        "kind": "n_s",
        "mean": 0.9648,
        "error": 0.0047,
    },
    "theta_star": {
        "kind": "100_theta_star",
        "mean": 1.04103,
        "error": 0.00026
    } 
}

def H0_to_theta(name_list, value_list):
    
    hubble = value_list[0]
    omega_nu = value_list[1]
    omnuh2 = value_list[2]
    mnu = value_list[3]
    TCMB = value_list[4]
    omega_m = value_list[5]
    omega_m_early = value_list[6]
    ommh2 = value_list[7]
    omega_c = value_list[8]
    omch2 = value_list[9]
    omega_b = value_list[10]
    ombh2 = value_list[11]
    omega_lambda = value_list[12]
    omlamh2 = value_list[13]
    omega_k = value_list[14]
    num_massive_neutrinos = value_list[15]
    nnu = value_list[16]
    w = value_list[17]
    wa = value_list[18]

    h = hubble / 100.
    if np.isnan(omnuh2):
        omnuh2 = omega_nu * h**2
    if np.isnan(omega_m):
        omega_m = ommh2 / h**2
    if np.isnan(ombh2):
        ombh2 = omega_b * h**2
    if np.isnan(omega_lambda):
        omega_lambda = omlamh2 / h**2
    if np.isnan(omch2):
        omch2 = omega_c * h**2
    if np.isnan(omega_m):
        omega_m = omega_c + omega_b + omega_nu
    if np.isnan(omega_m):
        omega_m = (omch2 + ombh2 + omnuh2) / h**2
    if np.isnan(omega_lambda):
        omega_lambda = 1 - omega_k - omega_m
    if np.isnan(mnu):
        mnu = omnuh2 / ((nnu / 3.0) ** 0.75 / 94.06410581217612 * (TCMB/2.7255)**3)
    if np.isnan([hubble, omega_m, omega_lambda, ombh2, mnu]).any():
        return np.nan

    original_feedback_level = camb.config.FeedbackLevel

    try:
        camb.set_feedback_level(0)
        p = camb.CAMBparams()
        p.set_cosmology(ombh2 = ombh2,
                        omch2 = omch2,
                        omk = omega_k,
                        mnu = mnu,
                        num_massive_neutrinos=num_massive_neutrinos,
                        nnu=nnu,
                        H0=hubble)
        p.set_dark_energy(w=w, wa=wa, dark_energy_model='ppf')
        r = camb.get_background(p)
        theta = r.cosmomc_theta()
    except:
        theta = np.nan
    finally:
        camb.config.FeedbackLevel = original_feedback_level
    return theta


def H0_to_theta_interface(block):
    
    name_list = ["h0", "omega_nu", "omnuh2", "mnu", "temp_cmb", "omega_m",
                    "omega_m_early", "ommh2", "omega_c", "omch2", "omega_b",
                    "ombh2", "omega_lambda", "omlamh2", "omega_k",
                    "num_massive_neutrinos", "nnu", "w", "wa"]

    value_list = []
    
    for name in name_list:
        # bad coding with try/except?
        try:
            value_list.append(block["cosmological_parameters", name])
        except:
            value_list.append(np.nan)


    return H0_to_theta(name_list, value_list)


class EarlyPlanckLikelihood(GaussianLikelihood):
    "Early Planck Data from Table 1 of https://arxiv.org/pdf/2302.12911v2"

    like_name = "early_Planck_Lemos"
   
    def __init__(self, options):
        data_sets = list(EARLY_PLANCK_DATA.keys())
        self.data_sets = data_sets
        super().__init__(options)


    def build_data(self):
        values = []
        kinds = []
        for name in self.data_sets:
            parameters = EARLY_PLANCK_DATA[name]
            values.append(parameters["value"])
            kinds.append(parameters["kind"])
        
        kinds = np.array(kinds)
        values = np.array(values)

        self.omegab_index = np.where(kinds=="omega_b")[0]
        self.omegac_index = np.where(kinds=="omega_c")[0]
        self.As_index = np.where(kinds=="1e9_A_s_exp_-2tau")[0]
        self.ns_index = np.where(kinds=="n_s")[0]
        self.theta_index = np.where(kinds=="100_theta_star")[0]

    def build_covariance(self):
        n = len(self.data_y)
        C = np.zeros((n, n))
        i = 0
        for name in self.data_sets:
            parameters = EARLY_PLANCK_DATA[name]
            C[i, i] = parameters["error"]**2
            i += 1
        return C
    
    def extract_theory_points(self, block):
        y = np.zeros[self.data_y.size]

        y[self.omegab_index] = block["cosmological_parameters",
                                     "omega_b"] * block["cosmological_parameters",
                                                        "h0"] ** 2
        y[self.omegac_index] = (block["cosmological_parameters",
                                      "omega_m_early"] - block["cosmological_parameters",
                                                         "omega_b"]) ** 2
        y[self.As_index] = block["cosmological_parameters",
                                 "A_s"] * 1e9 * np.exp(
            - 2 * block["cosmological_parameters", "tau"])
        y[self.ns_index] = block["cosmological_parameters", "n_s"]
        y[self.theta_index] = 100 * H0_to_theta_interface(block)



