---
title: '`SAJAX`: Stellar Activity grid for exoplanets in JAX'
authors:
  - name: Samson J. Mercier
    orcid: 0000-0002-0962-7585
    affiliation: 1
  - name: Lucas M. Arthur
    orcid: 0009-0002-9317-1169
    affiliation: 1
  - name: Hritam Chakraborty
    orcid: 0000-0002-5177-1898
    affiliation: 2
affiliations:
  - name: Department of Earth, Atmospheric and Planetary Sciences, Massachusetts Institute of Technology, 77 Massachusetts Avenue, Cambridge, MA 02139, USA
    index: 1
  - name: University of Warwick
    index: 2
date: 12 October 2026
bibliography: paper.bib

---

# Summary
When a planet passes in front of its host star, the resulting flux dip reveals the planet's size, orbit and, across wavelengths, its atmosphere. 
However, the stellar surface is not uniform: magnetic activity produces dark spots, bright faculae and short-lived flares that bias the measured transit depth, and leave their own spectral signatures. 
`SAJAX` is a Python package that simulates these effects. 
It places any number of active regions, each with its own spectrum and limb-darkening coefficients, on a pixelated grid of the visible stellar disk, rotates the star, and optionally transits one or more planets across it.
`SAJAX` offers active regions with tunable edge smoothness, from Gaussian to hard-edged, and properties that can evolve in time, capturing phenomena such as spot growth, latitudinal drift and flares. 
Written in `JAX` [@jax2018github], it returns spectroscopic light curves, runs on CPUs and GPUs, and provides exact gradients for modern inference tools.

# Statement of Need
The transit method has grown from a detection technique into a versatile tool for characterizing exoplanets and their atmospheres.
Early applications were limited by instrumental precision, a barrier that next-generation facilities such as the James Webb Space Telescope have largely lifted [@Rustamkulov2023]. 
At this new level of precision, our imperfect knowledge of host stars has become the primary source of systematic error in transit spectrophotometry.
Of the many forms of stellar contamination, the most impactful for transits are magnetic active regions, such as spots, faculae and flares.
Unocculted regions produce wavelength-dependent biases in the transit depth through the transit light source effect [@Rackham2017; @Rackham2018; @Rackham2019; @Rackham2023], while occulted ones lead to in-transit bumps that skew the measured depth if not modeled [@Murray2026]. 
The planet signal and stellar surface are degenerate and must therefore be modeled together to constrain the planet properties.
Conversely, because the transit chord scans the stellar disk, this joint modeling also turns each transit into a direct probe of the stellar surface.

`SAJAX` is designed for this joint problem: rather than treating stellar contamination as a correction applied after the fact, it fits the chromatic spot-crossing signature and planetary transmission spectrum simultaneously, so that each transit constrains both the planet and the active region(s) it crosses.
It provides a fast and flexible tool for exoplanet and stellar astronomers who need to (i) forward-model spectrophotometric time series of active stars, with or without transiting planets, and (ii) fit such models to data.


# State of the field
`SAJAX` joins an established ecosystem of codes that model the effect of stellar activity on transit light curves. 
These fall into three broad families, distinguished by how they represent active regions. 
*Analytic and semi-analytic* tools, such as `macula` [@Kipping2012], `spotrod` [@Beky2014], `STSP` [@Morris2017] and `fleck` [@Morris2020], describe circular active regions in closed or nearly closed form. 
*Spherical-harmonic* codes such as `starry` [@Luger2019], `starry_process` [@Luger2021], and `StarryStarryProcess` [@Sagynbayeva2025] expand the stellar surface map into spherical harmonics, reducing inference to fitting the expansion coefficients. 
This allows active regions of arbitrary shape, but it has some drawbacks: small regions require computationally demanding, high-degree terms; a truncated expansion can produce ringing effects; and certain combinations can produce negative intensities. 
Finally, *grid-based* codes tessellate the surface and assign properties to each cell; examples include `PyTranSpot` [@Juvan2018], `SAGE` [@Chakraborty2024], `spot-model` [@Cracchiolo2021a; @Cracchiolo2021b], and `spotter` [@Garcia2025]. 
The `SOAP` family [@Boisse2012; @Oshagh2013; @Dumusque2014; @Akinsanmi2018; @Serrano2020; @Cristo2025] and `ANTARESS` [@Bourrier2024] also model heterogeneous stellar surfaces, but with a primary focus on spectroscopic and radial-velocity observations.

The two codes closest to `SAJAX` are `fleck` and `spotter`, both of which offer `JAX` implementations.
`fleck` computes each active region's flux from its projected area and the limb-darkening at its center, and treats an active region as visible or hidden according to where its center lies on the visible hemisphere. 
This is efficient and accurate for small active regions, but breaks down for large ones, such as those found on active M dwarfs. 
`spotter` is built on a HEALPix map of the whole sphere [@Gorski2005]. It supports arbitrarily shaped active regions and Gaussian-process surface models. 

`SAJAX` extends `SAGE`, a grid-based `NumPy` code that assigns separate spectra to the quiet photosphere and spots but relies on Python loops and does not model planetary occultations. 
Its guiding philosophy is to be a lightweight, inference-focused model: each active region is described by a small set of physically interpretable parameters (position, size, edge smoothness, spectrum and limb-darkening) rather than a free-form surface map, and the model is differentiable and cheap.

In practice, `SAJAX` is the more natural choice when active regions are large or overlapping, where `fleck`'s point-source treatment breaks down. As a grid-based code `SAJAX` trades some of `fleck`'s analytical speed for flexibility, but remains fast enough for the thousands of evaluations inference requires. It shines when inferring a planet's transmission spectrum jointly with active regions, particularly when the latter require distinct treatment: e.g. their own limb-darkening law, an umbra nested in a penumbra, or successive crossings of regions at different temperatures.


# Software design
Here, we describe the key features that underpin `SAJAX`'s core philosophy.

**Two-stage interface** - Inference requires thousands of model evaluations or more, and `SAJAX` is built around this use case. 
The light curve model is split into two functions. 
`build_system` runs once in `NumPy` and performs the most expensive, parameter-independent work: building the stellar grid, limb-darkening angles, rotational velocity field, and, for a fixed orbit, planet positions. 
`make_lc` then takes the output of `build_system`, together with the active region, limb-darkening and (optionally) planetary parameters, and produces the light curve(s). 
As a pure `JAX` function, `make_lc` can be passed to `jit`, `vmap` and `grad`, making it ideal for inference software that parallelizes operations, leverages modern hardware, and benefits from gradient-based methods. `quick_lc` combines both steps for one-off use.

**Grid** - `SAJAX` keeps `SAGE`'s sky-plane Cartesian grid: a disk of radius $R$ pixels in which every pixel has the same projected area, so disk integration is a plain sum. 
Only the visible hemisphere is stored. 
We tested several grid parameterizations, and the sky-plane Cartesian grid was consistently the fastest: because its pixels are defined in the plane of the sky and already carry equal projected area, disk integration requires no re-projection of grid cells onto the sky plane.
It is also more economical than a HEALPix sphere, requiring half as many pixels for equal resolution at the disk center.
The cost is that resolution on the stellar surface degrades toward the limb.

**Active regions** - Each region is a super-Gaussian with respect to the spherical distance $\theta$ from its center,
$$S_a(\theta) = \exp\left[-\left(\frac{1-\cos\theta}{1-\cos r_a}\right)^{n_a}\right],$$
with angular radius $r_a$ and smoothness $n_a$. 
$n_a = 1$ gives a Gaussian, and $n_a \to \infty$ recovers the hard-edged parameterization. 
Because the profile is smooth for finite $n_a$, gradients with respect to position and size are well defined. 
Each surface component — the quiet photosphere, denoted q, or an active region $a$ — carries its own spectrum $F(\lambda)$ and its own limb-darkening law $L(\mu)$, normalized so that $2\int_0^1 L(\mu)\,\mu\,\mathrm{d}\mu = 1$.
The contribution of the pixel at $\mu_j$ is then
$$f_j(\lambda) = F_\mathrm{q}(\lambda)\,L_\mathrm{q}(\mu_j)\left[1 - \sum_a \left(1 - \frac{F_a(\lambda)\,L_a(\mu_j)}{F_\mathrm{q}(\lambda)\,L_\mathrm{q}(\mu_j)}\right) S_a\right],$$
and, because every pixel subtends the same projected area, the disk-integrated flux is the mean over the $N$ pixels of the disk,
$$F(\lambda) = \frac{1}{N}\sum_{j=1}^{N} f_j(\lambda),$$
with occulted pixels contributing zero.
The bracketed contrasts are dimensionless, so $F(\lambda)$ carries the units of the input spectra, and a spotless, untransited star returns $F_\mathrm{q}(\lambda)$.
Overlapping regions add, so an umbra inside a penumbra needs no special treatment. 
Because contrasts combine additively, a stack of strongly overlapping dark regions can in principle drive a pixel's contribution negative. To avoid this and ensure the surface remains physical, `SAJAX` clips it at zero.
Spectra are sampled at each pixel's rotationally Doppler-shifted wavelength.

**Planets** - Planets on Keplerian orbits occult pixels directly, so active region crossings follow naturally from the geometry. 
By default, each planet is a hard pixel mask, which blocks gradient flow for the orbital parameters. 
`SAJAX` offers an optional sigmoid edge that restores it, and recovers the transit depth of the hard-edged mask to fourth order.
Each planet also has a sky-projected spin–orbit angle, since repeated crossings of the same active region can constrain orbital geometry [@SanchisOjeda2011].

**Optional features at no cost** - Any active region property can carry a time axis, describing regions that grow, drift, brighten or dim. 
A brightening region models a flare, and `SAJAX` includes a built-in flare template based on @TovarMendoza2022. 
Multiple transiting planets are also supported. 
In keeping with `SAJAX`'s focus on inference, these features use separate code paths, so models that do not use them run no slower.

# Research impact statement 
`SAJAX` has not yet been used in a published study, but it is already being applied by the authors to model chromatic spot-crossing signatures in transit spectra of active stars and to study how the choice of Bayesian inference method affects the resulting parameters' constraints.
Beyond these ongoing applications, its expected significance rests on three points. 
First, it re-implements the pixel model of `SAGE`, which was developed to quantify stellar contamination of transmission spectra [@Chakraborty2024] and has been used to predict photometric variability of TOI-3884 [@Chakraborty2025]. `SAJAX` adds planet–spot crossings, gradients, and the potential for GPU execution, broadening its applicability.
Second, `SAJAX` is released via the Python Package Index (PyPI) under an MIT license, with versioned releases, automated CI and test infrastructure, documentation and tutorials on Read the Docs, and a public issue tracker, allowing for easy adoption by the community.
Finally, we provide examples of both forward modeling and inference. 
The inference example follows a well-established approach, fitting the white light curve for the orbit and spot geometry and then each wavelength channel of a chromatic transit with a spot crossing, with both a *maximum a posteriori* fit and Markov chain Monte Carlo, and compares the retrieved spot, orbit and transmission spectrum with the injected values.

# Future directions
`SAJAX` will be maintained as a lightweight, fast tool compatible with evolving inference software, and we are considering three potential extensions. 
The first, and most substantial, is producing radial-velocity time series, capturing the range of phenomena required of such models. 
Building this directly into `SAJAX` will reduce the cost of joint fits to transit and radial-velocity data (e.g. Rossiter-McLaughlin analyses): instead of two separate model evaluations per likelihood call, the light-curve and radial-velocity models could share a single pass over the grid built by `build_system`.
The second is modeling oblate planets. 
To our knowledge, no existing software computes the transit of an oblate planet across a star with active regions. 
JWST data are only beginning to constrain planetary oblateness [@Cassese2024; @Dholakia2025], and having a differentiable `JAX` framework for such cases would allow the information content of such observations to be explored directly.
The third is enabling transit-timing variations for multi-planet systems. 

# AI usage disclosure
The authors used Claude Code (versions 2.1.245–2.1.281, with Claude Sonnet 5) to assist with the initial port of `SAGE` to `JAX`, and the implementation of key features, unit tests, example notebooks and documentation. 
The authors made all design and scientific decisions, manually reviewed all changes, applied unit tests, and validated every feature against a set of benchmark cases. 
The authors take full responsibility for the software and the content of this manuscript.

# Acknowledgements
`SAJAX` builds on `SAGE` [@Chakraborty2024] and draws inspiration from `ANTARESS` [@Bourrier2024]. 
Its core depends on `JAX` [@jax2018github], `NumPy` [@Harris2020], `SciPy` [@Virtanen2020], `Matplotlib` [@Hunter2007] and `interpax`. 
The example notebooks additionally use `NumPyro` [@Phan2019], `ArviZ` [@Kumar2019], `corner` [@ForemanMackey2016], `ExoTiC-LD` [@Grant2024], and `speclib` [@Rackham2024].

We thank Julien de Wit, Benjamin V. Rackham, Catriona Murray, Zoe de Beurs, Prajwal Niraula and all other members of the Disruptive Planets team for fruitful discussions during the development of `SAJAX`. We additionally thank Benjamin V. Rackham for constructive feedback that strengthened this manuscript.
S.J.M. acknowledges support from the Massachusetts Institute of Technology through the Hurley Fellowship. 
L.M.A. acknowledges support from the Massachusetts Institute of Technology through the Lester Wolfe Fellowship and the MIT Presidential Fellowship.
The authors acknowledge the MIT Office of Research Computing and Data for providing high performance computing resources that have contributed to the research results reported within this paper.

# References