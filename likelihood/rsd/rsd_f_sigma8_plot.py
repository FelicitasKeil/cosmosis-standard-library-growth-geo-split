from cosmosis.gaussian_likelihood import GaussianLikelihood
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

# Data from Table 1 of https://arxiv.org/pdf/2205.05017

RSD_F_SIGMA_8_DATA = {
    "2MFT": {
        "z": 0.0,
        "f_sigma_8": 0.505,
        "error": 0.084,
    },
    "6dFGS": {
        "z": 0.067,
        "f_sigma_8": 0.423,
        "error": 0.055,
    },    
    "SDSS DR13": {
        "z": 0.1,
        "f_sigma_8": 0.48,
        "error": 0.16,
    },    
    "2dFGRS": {
        "z": 0.17,
        "f_sigma_8": 0.51,
        "error": 0.06,
    },    
    "GAMA 1": {
        "z": 0.2,
        "f_sigma_8": 0.43,
        "error": 0.05,
    },  
    "WiggleZ 1": {
        "z": 0.22,
        "f_sigma_8": 0.42,
        "error": 0.07,
    }, 
    "BOSS LOW Z 1": {
        "z": 0.25,
        "f_sigma_8": 0.471,
        "error": 0.024,
    },  
    "GAMA 2": {
        "z": 0.38,
        "f_sigma_8": 0.44,
        "error": 0.06,
    },  
    "BOOS LOW Z 2": {
        "z": 0.4,
        "f_sigma_8": 0.431,
        "error": 0.025,
    },  
    "WiggleZ 2": {
        "z": 0.41,
        "f_sigma_8": 0.45,
        "error": 0.04,
    },  
    "CMASS BOSS": {
        "z": 0.57,
        "f_sigma_8": 0.453,
        "error": 0.022,
    },  
    "WiggleZ 3": {
        "z": 0.6,
        "f_sigma_8": 0.43,
        "error": 0.04,
    },  
    "VIPERS 1": {
        "z": 0.6,
        "f_sigma_8": 0.48,
        "error": 0.12,
    },  
    "SDSS IV 1": {
        "z": 0.69,
        "f_sigma_8": 0.447,
        "error": 0.039,
    },  
    "SDSS IV 2": {
        "z": 0.77,
        "f_sigma_8": 0.432,
        "error": 0.038,
    },  
    "WiggleZ 4": {
        "z": 0.78,
        "f_sigma_8": 0.38,
        "error": 0.04,
    },  
    "SDSS IV 3": {
        "z": 0.85,
        "f_sigma_8": 0.52,
        "error": 0.10,
    },  
    "VIPERS 2": {
        "z": 0.86,
        "f_sigma_8": 0.48,
        "error": 0.10,
    },  
    "SDSS IV 4": {
        "z": 0.978,
        "f_sigma_8": 0.379,
        "error": 0.176,
    },  
    "SDSS IV 5": {
        "z": 1.23,
        "f_sigma_8": 0.385,
        "error": 0.1,
    },  
    "Fastsound": {
        "z": 1.4,
        "f_sigma_8": 0.494,
        "error": 0.123,
    },  
    "SDSS IV 6": {
        "z": 1.52,
        "f_sigma_8": 0.426,
        "error": 0.077,
    },  
    "SDSS IV 7": {
        "z": 1.944,
        "f_sigma_8": 0.364,
        "error": 0.106,
    },  
}

class RSDLikelihood(GaussianLikelihood):
    "RSD Data from Table 1 of https://arxiv.org/pdf/2205.05017"

    like_name = 'rsd_Blanchard'
    x_section = 'growth_parameters'
    x_name = 'z'
    y_section = 'growth_parameters'

    def __init__(self, options):
        data_sets = list(RSD_F_SIGMA_8_DATA.keys())
        self.data_sets = data_sets
        super().__init__(options)

    def build_data(self):
        z = []
        f_sigma_8 = []
        f_sigma_8_error = []
        for name in self.data_sets:
            surveys = RSD_F_SIGMA_8_DATA[name]

            # collect the redshfits for the measurements
            z.append(surveys["z"])
            f_sigma_8.append(surveys["f_sigma_8"])
            f_sigma_8_error.append(surveys["error"])

        z = np.array(z)
        f_sigma_8 = np.array(f_sigma_8)
        self.z = z
        self.f_sigma_8 = f_sigma_8
        self.f_sigma_8_error = np.array(f_sigma_8_error)

        return z, f_sigma_8
    
    def build_covariance(self):
        n = len(self.data_x)
        C = np.zeros((n, n))
        i = 0
        for name in self.data_sets:
            surveys = RSD_F_SIGMA_8_DATA[name]
            C[i, i] = surveys["error"]**2
            i += 1
        return C
    
    def extract_theory_points(self, block):
        # get f * sigma 8 from the Einstein-Boltzmann solver
        z_theory = block[self.x_section, self.x_name]
        f_sigma8 = block[self.y_section, "fsigma_8_growth"]
        z_data = self.data_x
        f = scipy.interpolate.interp1d(z_theory, f_sigma8)
        y = f(z_data)

        plt.plot(z_data, y, label = 'Theory prediction w. Planck 2018')
        plt.errorbar(self.z, self.f_sigma_8, self.f_sigma_8_error, fmt='o',
                     label='RSD values')
        plt.xlabel('z')
        plt.ylabel(r'f $\sigma_8$(z)')
        plt.title(r'f $\sigma_8$(z) Cosmosis Likelihood')
        plt.legend()
        plt.savefig('f_sigma_8_theory_data.png')

        return y

setup, execute, cleanup = RSDLikelihood.build_module()