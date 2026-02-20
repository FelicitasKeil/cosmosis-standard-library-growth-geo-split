from numpy import log, pi, interp, loadtxt, dot, append, linalg, genfromtxt, sqrt
import os
from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section
from cosmosis.gaussian_likelihood import GaussianLikelihood
import matplotlib.pyplot as plt
from scipy import interpolate

cosmo = section_names.cosmological_parameters
likes = section_names.likelihoods
growthparams = section_names.growth_parameters
dist = section_names.distances

ROOT_dir = os.path.split(os.path.abspath(__file__))[0]

c_km_per_s = 299792.458
default_rd_fiducial = 147.8


class MGSLikelihood(GaussianLikelihood):
	
	data_type = "MGS"
	like_name = "mgs"
	def __init__(self, options):
		
		super(MGSLikelihood, self).__init__(options)
		# Allow override of these parameters
		self.rd_fiducial = self.options.get_double("rd_fiducial", default_rd_fiducial)
		self.feedback = self.options.get_bool("feedback", default=False)
		
	def build_data(self):
		
		print("LRG+MGS data")
		print("FS: f(z)sigma8(z)")
		# Reading data file
		DATA_file = os.path.join(ROOT_dir,
						   "sdss_LRG_MGS_QSO_ELG_GAL_FS_fs8.txt")
			
		DATA = loadtxt(DATA_file, usecols=(0, 1))
		z_eff, data = DATA[:, 0], DATA[:, 1]
		print('data', data)
		
		return z_eff, data
		
	def build_covariance(self):
		
		# Reading covariance matrix file
		COV_file = os.path.join(ROOT_dir,
						  'sdss_LRG_MGS_QSO_ELG_GAL_FS_fs8_covtot.txt')
			
		cov = loadtxt(COV_file)
		self.inv_cov = linalg.inv(cov)
		
		return cov
		
	def build_inverse_covariance(self):
		return self.inv_cov
		
	def extract_theory_points(self, block):
		
		# Redshift array
		z = block[dist, 'z']
		
		#Find theory Dm and Dh at effective redshift by interpolation
		z_eff = self.data_x
		
		z = block['growth_parameters', 'z']
		fsigma8 = block['growth_parameters', 'fsigma_8']
		# Find theory fsigma8 at fiducial redshift
		fsigma8_interp = interpolate.interp1d(z, fsigma8)
		fsigma8_z = fsigma8_interp(z_eff)
		print('f s8 theory', fsigma8_z)

		plt.plot(z, fsigma8_interp(z), label = 'Theory prediction')
		plt.errorbar(z_eff[0], self.data_y[0],
			   sqrt(self.build_covariance()[0, 0]),
			   fmt='o', label='MGS')
		plt.errorbar(z_eff[1], self.data_y[1],
			   sqrt(self.build_covariance()[1, 1]),
			   fmt='o', label='BOSS Galaxy 0.4-0.6')
		plt.errorbar(z_eff[2], self.data_y[2],
			   sqrt(self.build_covariance()[2, 2]),
			   fmt='o', label='LRG')
		plt.errorbar(z_eff[3], self.data_y[3],
			   sqrt(self.build_covariance()[3, 3]),
			   fmt='o', label='ELG')
		plt.errorbar(z_eff[4], self.data_y[4],
			   sqrt(self.build_covariance()[4, 4]),
			   fmt='o', label='QSO')

		plt.xlabel('z')
		plt.ylabel(r'f $\sigma_8$(z)')
		plt.title(r'f $\sigma_8$(z) Cosmosis Likelihood')
		plt.legend()
		plt.savefig('f_sigma_8_theory_data_eboss.png')
		
		if self.feedback:
			print()
			print('             zeff   pred    data')
			print('fsigma8:', self.data_x, fsigma8_z, self.data_y)
			print()
			
		return fsigma8_z
		
setup, execute, cleanup = MGSLikelihood.build_module()
