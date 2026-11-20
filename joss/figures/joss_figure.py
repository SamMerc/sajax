#!/usr/bin/env python
# coding: utf-8
"""
Make the paper's main figure (main_plot.pdf) from simulated_data.npz, stage1_nuts.npz and
stage2_nuts.npz in --results-dir (default: this script's directory). Needs numpy + matplotlib.
"""
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent)
args = p.parse_args()

# Non-interactive backend; must be set before importing pyplot
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

def wavelength_colormap(wavelength, cmap='viridis_r', trim=(0.0, 1.0)):
    cmap = ListedColormap(plt.get_cmap(cmap)(np.linspace(*trim, len(wavelength))))
    edges = np.concatenate([
        [1.5 * wavelength[0] - 0.5 * wavelength[1]],
        (wavelength[:-1] + wavelength[1:]) / 2.0,
        [1.5 * wavelength[-1] - 0.5 * wavelength[-2]],
    ])
    return cmap, BoundaryNorm(edges, cmap.N)

RESULTS_DIR = args.results_dir

# ---- Inputs ----
_sim = np.load(RESULTS_DIR / "simulated_data.npz")
wavelength    = _sim["wavelength"]
spec_quiet    = _sim["spec_quiet"]
spec_penumbra = _sim["spec_penumbra"]
spec_umbra    = _sim["spec_umbra"]
k_wavelength  = _sim["k_wavelength"]
times         = _sim["times"]
lc_white_obs  = _sim["lc_white_obs"]
lc_2d_obs     = _sim["lc_2d_obs"]
NWAVE         = len(wavelength)

_stage1 = np.load(RESULTS_DIR / "stage1_nuts.npz")
lc_bestfit_white        = _stage1["lc_bestfit_white"]
residuals_bestfit_white = _stage1["residuals_bestfit_white"]
star_maps_bestfit       = _stage1["star_maps_bestfit"]
vmin_bf, vmax_bf        = float(star_maps_bestfit.min()), float(star_maps_bestfit.max())

_stage2 = np.load(RESULTS_DIR / "stage2_nuts.npz")
lc_bestfit_2d = _stage2["lc_bestfit_2d"]
residuals_2d  = _stage2["residuals_2d"]
samples_2d    = {key.removeprefix("sample_"): _stage2[key] for key in _stage2.files if key.startswith("sample_")}

wave_cmap, wave_norm = wavelength_colormap(wavelength, cmap='turbo', trim=(0.1, 0.9))


fig, ax = plt.subplot_mosaic([['system_view_cbar', 'light_curve', 'residual_histogram'],

                                ['system_view', 'light_curve', 'residual_histogram'],

                                ['penumbra_spectrum', 'light_curve', 'residual_histogram'],

                                ['umbra_spectrum', 'transmission_spectrum', '.'],],
                                figsize=(15, 10), height_ratios=[0.2, 8., 5, 5], width_ratios=[1, 2, 0.5], gridspec_kw={'wspace':0.25, 'hspace':0.3})

# System view
im = ax['system_view'].imshow(star_maps_bestfit[250], cmap='magma', origin='lower', vmin=vmin_bf, vmax=vmax_bf, aspect='auto')
ax['system_view'].set_xticks([]); ax['system_view'].set_yticks([])
cbar = fig.colorbar(im, cax=ax['system_view_cbar'], orientation='horizontal')
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.set_label('Normalized Flux')

# Light curve
## WLC
ax['light_curve'].plot(times, lc_white_obs, '.', ms=3, alpha=0.5, color='gray')
ax['light_curve'].plot(times, lc_bestfit_white, 'k-', lw=1.)
## CLCs
for i in range(NWAVE):
    color = wave_cmap(wave_norm(wavelength[i]))
    if i == 0:
        ax['light_curve'].plot(times, lc_2d_obs[:, i] - (i+1)*0.004, '.', ms=3, alpha=0.5, color=color)
        ax['light_curve'].plot(times, lc_bestfit_2d[:, i] - (i+1)*0.004, 'k-', lw=1.)
    else:
        ax['light_curve'].plot(times, lc_2d_obs[:, i] - (i+1)*0.008, '.', ms=3, alpha=0.5, color=color)
        ax['light_curve'].plot(times, lc_bestfit_2d[:, i] - (i+1)*0.008, 'k-', lw=1.)
ax['light_curve'].set_xlabel('Time [days]')
ax['light_curve'].set_ylabel('Normalized Flux + Offset')

# Residual histogram
CHANNEL_SPACING = 0.008
TARGET_FRACTION = 0.7   # how much of the per-channel slot the histogram should fill
RESID_VSCALE = TARGET_FRACTION * CHANNEL_SPACING / (4 * np.std(residuals_2d))
BLUE_HIST_SHIFT = 0.0015  # nudge the first (bluest) histogram down so it clears the white-light one

ax['residual_histogram'].sharey(ax['light_curve'])
ax['residual_histogram'].hist(
    RESID_VSCALE * (residuals_bestfit_white) + lc_bestfit_white[-1],
    bins=30, orientation='horizontal', color='lightgray',
)
for i in range(NWAVE):
    color = wave_cmap(wave_norm(wavelength[i]))
    if i == 0: baseline = lc_bestfit_2d[-40, i] - (i+1)*0.004 - BLUE_HIST_SHIFT
    else:      baseline = lc_bestfit_2d[-1, i] - (i+1)*0.008
    ax['residual_histogram'].hist(
        RESID_VSCALE * residuals_2d[:, i] + baseline,
        bins=30, orientation='horizontal', color=color,
    )
ax['residual_histogram'].set_xlim(left=0)
ax['residual_histogram'].axis('off')

# Contrast spectrum
## Umbra
for i in range(NWAVE):
    color = wave_cmap(wave_norm(wavelength[i]))
    ax['umbra_spectrum'].errorbar(wavelength[i], np.median(np.array(samples_2d["ar_contrast_umbra"]), axis=0)[i], fmt='.',
                                        yerr=(np.percentile(np.array(samples_2d["ar_contrast_umbra"]), 75, axis=0)[i]-np.percentile(np.array(samples_2d["ar_contrast_umbra"]), 25, axis=0)[i])/2,
                                        color=color, markersize=12)
ax['umbra_spectrum'].plot(wavelength, spec_umbra / spec_quiet, '-', color='black', lw=1.)
ax['umbra_spectrum'].set_xlabel('Wavelength [$\\AA$]')
ax['umbra_spectrum'].set_ylabel('Contrast Spectrum')
ax['umbra_spectrum'].text(0.75, 0.05, 'Umbra', transform=ax['umbra_spectrum'].transAxes, fontsize=12)

## Penumbra
for i in range(NWAVE):
    color = wave_cmap(wave_norm(wavelength[i]))
    ax['penumbra_spectrum'].errorbar(wavelength[i], np.median(np.array(samples_2d["ar_contrast_penumbra"]), axis=0)[i], fmt='.',
                                        yerr=(np.percentile(np.array(samples_2d["ar_contrast_penumbra"]), 75, axis=0)[i]-np.percentile(np.array(samples_2d["ar_contrast_penumbra"]), 25, axis=0)[i])/2,
                                        color=color, markersize=12)
ax['penumbra_spectrum'].plot(wavelength, spec_penumbra / spec_quiet, '-', color='black', lw=1.)
ax['penumbra_spectrum'].set_ylabel('Contrast Spectrum')
ax['penumbra_spectrum'].set_xticklabels([])
ax['penumbra_spectrum'].text(0.65, 0.05, 'Penumbra', transform=ax['penumbra_spectrum'].transAxes, fontsize=12)

# Transmission spectrum
for i in range(NWAVE):
    color = wave_cmap(wave_norm(wavelength[i]))
    ax['transmission_spectrum'].errorbar(wavelength[i], np.median(np.array(samples_2d["k"]), axis=0)[i], fmt='.',
                                        yerr=(np.percentile(np.array(samples_2d["k"]), 75, axis=0)[i]-np.percentile(np.array(samples_2d["k"]), 25, axis=0)[i])/2,
                                        color=color, markersize=12)
ax['transmission_spectrum'].plot(wavelength, k_wavelength, '-', color='black', lw=1.)
ax['transmission_spectrum'].set_ylabel('R$_p$/R$_*$')
ax['transmission_spectrum'].set_xlabel('Wavelength [$\\AA$]')


# Move the histograms closer to the light curves than wspace allows (done last so nothing overrides it)
pos_lc   = ax['light_curve'].get_position()
pos_hist = ax['residual_histogram'].get_position()

GAP = 0.01   # figure-fraction units
new_x0 = pos_lc.x0 + pos_lc.width + GAP
ax['residual_histogram'].set_position([new_x0, pos_hist.y0, pos_hist.width, pos_hist.height])

# Panel letters, in reading order, in each panel's upper-left corner
PANEL_LETTERS = {
    'system_view':           'A',
    'light_curve':           'B',   # the residual histograms belong to this panel
    'penumbra_spectrum':     'C',
    'umbra_spectrum':        'D',
    'transmission_spectrum': 'E',
}
for name, letter in PANEL_LETTERS.items():
    ax[name].text(0.03, 0.97, letter, transform=ax[name].transAxes, fontsize=16, fontweight='bold',
                  va='top', ha='left', color='white' if name == 'system_view' else 'black')   # A sits on the dark sky

plt.savefig(RESULTS_DIR / 'main_plot.pdf')
print(f"Saved {RESULTS_DIR / 'main_plot.pdf'}")
