from __future__ import print_function
from builtins import range
from cosmosis.datablock import option_section, names
import matplotlib.pyplot as plt
import pickle
import numpy as np

def setup(options):
    do_shear_shear = options.get_bool(option_section, "shear-shear", True)
    do_position_shear = options.get_bool(
        option_section, "position-shear", True)
    do_shear_cmbkappa=options.get_bool(option_section,"shear-cmbkappa",False)
    perbin = options.get_bool(option_section, "perbin", False)

    suffix = options.get_string(option_section, "suffix", "")

    print()
    print("The add_intrinsic module will try to combine IA terms into these spectra:")
    if do_shear_shear:
        print(" - shear-shear.")
    if do_position_shear:
        print(" - position-shear.")
    if do_shear_cmbkappa:
        print(" - shear-CMB kappa ")
    if not (do_shear_cmbkappa or do_position_shear or do_shear_shear):
        print("... actually not into anything. You set shear-shear=F and position-shear=F  shear-cmbkappa=F")
        print("Ths module will not do anything in this configuration")
    print()

    if suffix:
        suffix = "_" + suffix

    sec_names = {
        "shear_shear": "shear_cl" + suffix,
        "shear_shear_bb": "shear_cl_bb" + suffix,
        "shear_shear_gg": "shear_cl_gg" + suffix,
        "galaxy_shear": "galaxy_shear_cl" + suffix,
        "shear_intrinsic": "shear_cl_gi" + suffix,
        "galaxy_intrinsic": "galaxy_intrinsic_cl"  + suffix,
        "intrinsic_intrinsic": "shear_cl_ii"  + suffix,
        "intrinsic_intrinsic_bb": "shear_cl_ii_bb"  + suffix,
        "parameters": "intrinsic_alignment_parameters" + suffix,
        "shear_cmbkappa": "shear_cmbkappa_cl" + suffix,
        "intrinsic_cmbkappa": "intrinsic_cmbkappa_cl" + suffix,
    }   

    return do_shear_shear, do_position_shear, do_shear_cmbkappa, perbin, sec_names


def execute(block, config):
    do_shear_shear, do_position_shear, do_shear_cmbkappa, perbin, sec_names = config

    shear_shear = sec_names['shear_shear']
    shear_shear_bb = sec_names['shear_shear_bb']
    shear_shear_gg = sec_names['shear_shear_gg']
    galaxy_shear = sec_names['galaxy_shear']
    galaxy_intrinsic = sec_names['galaxy_intrinsic']
    shear_intrinsic = sec_names['shear_intrinsic']
    parameters = sec_names['parameters']
    intrinsic_intrinsic = sec_names['intrinsic_intrinsic']
    intrinsic_intrinsic_bb = sec_names['intrinsic_intrinsic_bb']
    shear_cmbkappa = sec_names['shear_cmbkappa']
    intrinsic_cmbkappa = sec_names['intrinsic_cmbkappa']

    if do_shear_shear:
        nbin_shear = block[shear_shear, 'nbin']
    elif do_position_shear:
        nbin_shear = block[galaxy_intrinsic, 'nbin_b']
    elif do_shear_cmbkappa:
        nbin_shear = block[shear_cmbkappa, 'nbin_a']
    if do_position_shear:
        nbin_pos = block[galaxy_shear, 'nbin_a']

    if perbin:
        A = [block[parameters, "A{}".format(i + 1)]
             for i in range(nbin_shear)]
    else:
        A = [1 for i in range(nbin_shear)]

    if do_shear_shear:
        # for shear-shear, we're replacing 'shear_cl' (the GG term) with GG+GI+II...
        # so in case useful, save the GG term to shear_cl_gg.
        # also check for a b-mode contribution from IAs
        block[shear_shear_gg, 'ell'] = block[shear_shear, 'ell']

        # Add metadata to the backup gg-only section
        for key in ["nbin_a", "nbin_b", "nbin", "sample_a", "sample_b", "is_auto", "auto_only", "sep_name"]:
            if block.has_value(shear_shear, key):
                block[shear_shear_gg, key] = block[shear_shear, key]

        for i in range(nbin_shear):
            for j in range(i + 1):
                bin_ij = 'bin_{0}_{1}'.format(i + 1, j + 1)
                bin_ji = 'bin_{1}_{0}'.format(i + 1, j + 1)
                block[shear_shear_gg, bin_ij] = block[shear_shear, bin_ij]
                block[shear_shear, bin_ij] += (
                    A[i] * A[j] * block[intrinsic_intrinsic, bin_ij]  # II
                    + A[j] * block[shear_intrinsic, bin_ij]  # The two GI terms
                    + A[i] * block[shear_intrinsic, bin_ji]
                )
                if block.has_section(intrinsic_intrinsic_bb):
                    block[shear_shear_bb, bin_ij] = block[intrinsic_intrinsic_bb, bin_ij]

    if do_position_shear:
        for i in range(nbin_pos):
            for j in range(nbin_shear):
                bin_ij = 'bin_{0}_{1}'.format(i + 1, j + 1)
                block[galaxy_shear, bin_ij] += A[j] * \
                    block[galaxy_intrinsic, bin_ij]

    if do_shear_cmbkappa:
        for i in range(nbin_shear):
            bin_ij = 'bin_{0}_{1}'.format(i + 1, 1)
            block[shear_cmbkappa, bin_ij] += A[i] * \
                    block[intrinsic_cmbkappa, bin_ij]
    
    # C_ELL FOR WEAK LENSING
    # with open("C_ell_shear_om-geo-03-growth-035-early-03.pkl", "wb") as out_pkl:
    #     pickle.dump(block[shear_shear, "bin_4_4"], out_pkl)

    with open("C_ell_shear_om-geo-03-growth-03-early-03.pkl", "rb") as in_pkl:
        C_ell_shear_geo_03_gr_03_ear_03 = pickle.load(in_pkl)
    
    with open("C_ell_shear_om-geo-03-growth-025-early-03.pkl", "rb") as in_pkl:
        C_ell_shear_geo_03_gr_025_ear_03 = pickle.load(in_pkl)

    with open("C_ell_shear_om-geo-03-growth-035-early-03.pkl", "rb") as in_pkl:
        C_ell_shear_geo_03_gr_035_ear_03 = pickle.load(in_pkl)

    with open("C_ell_shear_om-geo-025-growth-03-early-03.pkl", "rb") as in_pkl:
        C_ell_shear_geo_025_gr_03_ear_03 = pickle.load(in_pkl)

    with open("C_ell_shear_om-geo-035-growth-03-early-03.pkl", "rb") as in_pkl:
        C_ell_shear_geo_035_gr_03_ear_03 = pickle.load(in_pkl)

    with open("C_ell_shear_om-geo-03-growth-03-early-025.pkl", "rb") as in_pkl:
        C_ell_shear_geo_03_gr_03_ear_025 = pickle.load(in_pkl)

    with open("C_ell_shear_om-geo-03-growth-03-early-035.pkl", "rb") as in_pkl:
        C_ell_shear_geo_03_gr_03_ear_035 = pickle.load(in_pkl)

    plt.plot(block[shear_shear, 'ell'], block[shear_shear, 'ell'] * (
        2 * block[shear_shear, 'ell'] + 1) * C_ell_shear_geo_03_gr_03_ear_03,
        label="$\\Omega_{\\rm m}=0.3$", color="black")
    plt.plot(block[shear_shear, 'ell'], block[shear_shear, 'ell'] * (
        2 * block[shear_shear, 'ell'] + 1) * C_ell_shear_geo_03_gr_025_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.25, \\Omega^{early}_{\\rm m}=0.3$")
    plt.plot(block[shear_shear, 'ell'], block[shear_shear, 'ell'] * (
        2 * block[shear_shear, 'ell'] + 1) * C_ell_shear_geo_03_gr_035_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.35, \\Omega^{early}_{\\rm m}=0.3$")
    plt.plot(block[shear_shear, 'ell'], block[shear_shear, 'ell'] * (
        2 * block[shear_shear, 'ell'] + 1) * C_ell_shear_geo_025_gr_03_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.25, \\Omega^{growth}_{\\rm m}=0.3, \\Omega^{early}_{\\rm m}=0.3$")
    plt.plot(block[shear_shear, 'ell'], block[shear_shear, 'ell'] * (
        2 * block[shear_shear, 'ell'] + 1) * C_ell_shear_geo_035_gr_03_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.35, \\Omega^{growth}_{\\rm m}=0.3, \\Omega^{early}_{\\rm m}=0.3$")
    plt.plot(block[shear_shear, 'ell'], block[shear_shear, 'ell'] * (
        2 * block[shear_shear, 'ell'] + 1) * C_ell_shear_geo_03_gr_03_ear_025,
        label="$\\Omega^{geo}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.3, \\Omega^{early}_{\\rm m}=0.25$")
    plt.plot(block[shear_shear, 'ell'], block[shear_shear, 'ell'] * (
        2 * block[shear_shear, 'ell'] + 1) * C_ell_shear_geo_03_gr_03_ear_035,
        label="$\\Omega^{geo}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.3, \\Omega^{early}_{\\rm m}=0.35$")
    plt.title(r'Weak Lensing $C(\ell)$, bin 4-4')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(2\ell+1)C^{\rm LL}_{4,4}(\ell)$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1, 1e5)
    # plt.ylim(1e-6, 1e-2)
    plt.legend(prop={'size': 7})
    plt.savefig("C_ell_shear_shear_split_early_cut.png", dpi=300)
    plt.clf()

    
    plt.hlines(1, np.min(block[shear_shear, 'ell']),
               np.max(block[shear_shear, 'ell']),
               label="$\\Omega_{\\rm m}=0.3$", color="black")
    plt.plot(block[shear_shear, 'ell'],
             C_ell_shear_geo_03_gr_025_ear_03 / C_ell_shear_geo_03_gr_03_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.25, \\Omega^{early}_{\\rm m}=0.3$")
    plt.plot(block[shear_shear, 'ell'], 
             C_ell_shear_geo_03_gr_035_ear_03 / C_ell_shear_geo_03_gr_03_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.35, \\Omega^{early}_{\\rm m}=0.3$")
    plt.plot(block[shear_shear, 'ell'],
             C_ell_shear_geo_025_gr_03_ear_03 / C_ell_shear_geo_03_gr_03_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.25, \\Omega^{growth}_{\\rm m}=0.3, \\Omega^{early}_{\\rm m}=0.3$")
    plt.plot(block[shear_shear, 'ell'], 
             C_ell_shear_geo_035_gr_03_ear_03  / C_ell_shear_geo_03_gr_03_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.35, \\Omega^{growth}_{\\rm m}=0.3, \\Omega^{early}_{\\rm m}=0.3$")
    plt.plot(block[shear_shear, 'ell'], 
             C_ell_shear_geo_03_gr_03_ear_025 / C_ell_shear_geo_03_gr_03_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.3, \\Omega^{early}_{\\rm m}=0.25$")
    plt.plot(block[shear_shear, 'ell'], 
             C_ell_shear_geo_03_gr_03_ear_035 / C_ell_shear_geo_03_gr_03_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.3, \\Omega^{early}_{\\rm m}=0.35$")
    plt.title(r'Weak Lensing $C(\ell)$ Ratios wrt. $\Omega_{\rm m}=0.3$, bin 4-4')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C^{\rm LL}_{\rm 4,4,rescaled} (\ell)/C^{\rm LL}_{4,4}(\ell)$')
    plt.xscale('log')
    plt.xlim(1, 1e5)
    plt.legend(prop={'size': 7})
    plt.savefig("C_ell_shear_shear_split_early_ratio_cut.png", dpi=300)
    plt.clf()

    # C_ELL FOR GALAXY CLUSTERING
    # with open("C_ell_clustering_om-geo-03-growth-03-early-035.pkl", "wb") as out_pkl:
    #     pickle.dump(block[names.galaxy_cl, "bin_1_1"], out_pkl)

    with open("C_ell_clustering_om-geo-03-growth-03-early-03.pkl", "rb") as in_pkl:
        C_ell_clustering_geo_03_gr_03_ear_03 = pickle.load(in_pkl)
    
    with open("C_ell_clustering_om-geo-03-growth-025-early-03.pkl", "rb") as in_pkl:
        C_ell_clustering_geo_03_gr_025_ear_03 = pickle.load(in_pkl)

    with open("C_ell_clustering_om-geo-03-growth-035-early-03.pkl", "rb") as in_pkl:
        C_ell_clustering_geo_03_gr_035_ear_03 = pickle.load(in_pkl)

    with open("C_ell_clustering_om-geo-025-growth-03-early-03.pkl", "rb") as in_pkl:
        C_ell_clustering_geo_025_gr_03_ear_03 = pickle.load(in_pkl)

    with open("C_ell_clustering_om-geo-035-growth-03-early-03.pkl", "rb") as in_pkl:
        C_ell_clustering_geo_035_gr_03_ear_03 = pickle.load(in_pkl)

    with open("C_ell_clustering_om-geo-03-growth-03-early-025.pkl", "rb") as in_pkl:
        C_ell_clustering_geo_03_gr_03_ear_025 = pickle.load(in_pkl)

    with open("C_ell_clustering_om-geo-03-growth-03-early-035.pkl", "rb") as in_pkl:
        C_ell_clustering_geo_03_gr_03_ear_035 = pickle.load(in_pkl)

    plt.plot(block[names.galaxy_cl, 'ell'], block[names.galaxy_cl, 'ell'] * (
        2 * block[names.galaxy_cl, 'ell'] + 1) * C_ell_clustering_geo_03_gr_03_ear_03,
        label="$\\Omega_{\\rm m}=0.3$", color="black")
    plt.plot(block[names.galaxy_cl, 'ell'], block[names.galaxy_cl, 'ell'] * (
        2 * block[names.galaxy_cl, 'ell'] + 1) * C_ell_clustering_geo_03_gr_025_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.25, \\Omega^{early}_{\\rm m}=0.3$")
    plt.plot(block[names.galaxy_cl, 'ell'], block[names.galaxy_cl, 'ell'] * (
        2 * block[names.galaxy_cl, 'ell'] + 1) * C_ell_clustering_geo_03_gr_035_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.35, \\Omega^{early}_{\\rm m}=0.3$")
    plt.plot(block[names.galaxy_cl, 'ell'], block[names.galaxy_cl, 'ell'] * (
        2 * block[names.galaxy_cl, 'ell'] + 1) * C_ell_clustering_geo_025_gr_03_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.25, \\Omega^{growth}_{\\rm m}=0.3, \\Omega^{early}_{\\rm m}=0.3$")
    plt.plot(block[names.galaxy_cl, 'ell'], block[names.galaxy_cl, 'ell'] * (
        2 * block[names.galaxy_cl, 'ell'] + 1) * C_ell_clustering_geo_035_gr_03_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.35, \\Omega^{growth}_{\\rm m}=0.3, \\Omega^{early}_{\\rm m}=0.3$")
    plt.plot(block[names.galaxy_cl, 'ell'], block[names.galaxy_cl, 'ell'] * (
        2 * block[names.galaxy_cl, 'ell'] + 1) * C_ell_clustering_geo_03_gr_03_ear_025,
        label="$\\Omega^{geo}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.3, \\Omega^{early}_{\\rm m}=0.25$")
    plt.plot(block[names.galaxy_cl, 'ell'], block[names.galaxy_cl, 'ell'] * (
        2 * block[names.galaxy_cl, 'ell'] + 1) * C_ell_clustering_geo_03_gr_03_ear_035,
        label="$\\Omega^{geo}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.3, \\Omega^{early}_{\\rm m}=0.35$")
    plt.title(r'Galaxy Clustering $C(\ell)$, bin 1-1')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(2\ell+1)C^{\rm GG}_{1,1}(\ell)$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(prop={'size': 7})
    plt.savefig("C_ell_clustering_split_early.png", dpi=300)
    plt.clf()

    
    plt.hlines(1, np.min(block[names.galaxy_cl, 'ell']),
               np.max(block[names.galaxy_cl, 'ell']),
               label="$\\Omega_{\\rm m}=0.3$", color="black")
    plt.plot(block[names.galaxy_cl, 'ell'],
             C_ell_clustering_geo_03_gr_025_ear_03 / C_ell_clustering_geo_03_gr_03_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.25, \\Omega^{early}_{\\rm m}=0.3$")
    plt.plot(block[names.galaxy_cl, 'ell'], 
             C_ell_clustering_geo_03_gr_035_ear_03 / C_ell_clustering_geo_03_gr_03_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.35, \\Omega^{early}_{\\rm m}=0.3$")
    plt.plot(block[names.galaxy_cl, 'ell'],
             C_ell_clustering_geo_025_gr_03_ear_03 / C_ell_clustering_geo_03_gr_03_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.25, \\Omega^{growth}_{\\rm m}=0.3, \\Omega^{early}_{\\rm m}=0.3$")
    plt.plot(block[names.galaxy_cl, 'ell'], 
             C_ell_clustering_geo_035_gr_03_ear_03  / C_ell_clustering_geo_03_gr_03_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.35, \\Omega^{growth}_{\\rm m}=0.3, \\Omega^{early}_{\\rm m}=0.3$")
    plt.plot(block[names.galaxy_cl, 'ell'], 
             C_ell_clustering_geo_03_gr_03_ear_025 / C_ell_clustering_geo_03_gr_03_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.3, \\Omega^{early}_{\\rm m}=0.25$")
    plt.plot(block[names.galaxy_cl, 'ell'], 
             C_ell_clustering_geo_03_gr_03_ear_035 / C_ell_clustering_geo_03_gr_03_ear_03,
        label="$\\Omega^{geo}_{\\rm m}=0.3, \\Omega^{growth}_{\\rm m}=0.3, \\Omega^{early}_{\\rm m}=0.35$")
    plt.title(r'Galaxy Clustering $C(\ell)$ Ratios wrt. $\Omega_{\rm m}=0.3$, bin 1-1')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C^{\rm GG}_{\rm 1,1,rescaled} (\ell)/C^{\rm GG}_{1,1}(\ell)$')
    plt.xscale('log')
    plt.legend(prop={'size': 7})
    plt.savefig("C_ell_clustering_split_early_ratio.png", dpi=300)
    plt.clf()

    return 0
