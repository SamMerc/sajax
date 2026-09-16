---
title: '`SAJAX`: Stellar Activity Grid for Exoplanets in JAX'
tags:
authors:
  - name: Samson J. Mercier
    orcid: 0000-0002-0962-7585
    affiliation: 1
  - name: Lucas Arthur
    orcid: 0009-0002-9317-1169
    affiliation: 1
affiliations:
  - name: Department of Earth, Atmospheric and Planetary Sciences, Massachusetts Institute of Technology, 77 Massachusetts Avenue, Cambridge, MA 02139, USA
    index: 1
date: XXXXXX
bibliography: paper.bib

---

# Summary

The transit method is a ubiquitous method for detecting exoplanets and provides key opportunities to charracterizes exoplanets and their weird characteristics. Conversely, this method also gives us a window in the stars these worlds orbit: by passing across the stellar disk, we can probe the local stellar surface occulted during the transit chord. These two sides of exoplanet transits are intertwined and must be modeled synchronously to draw precise and accurate conclusions on the properties of these distant systems. An important piece of the puzzle is stellar activity and in particular, magnetic active regions:  variable magnetic activity across the stellar surface can lead to the birth of dark or bright regions (commonly referred to as spots and faculae). These regions play an important role in shaping the stellar disk crossed by the transit and must therefore be modeled. Sajax represents a novel framework in modeling these active regions and their impact on exoplanet transits. 

# Statement of Need
xxxxx
# Documentation

Documentation for SAJAX is available at https://sajax.readthedocs.io/en/latest/.

# Similar Tools and Comparison

While sajax is by no means the only such transit modeling tool out there, we are indeed standing on the shoulders of giants that came before us, it provides some key features that place it at the forefront of the field and could combine seamlessly with the stream of high-quality data delivered by next-generation telescopes.

It is important for us to first adress the list of tools that are out there and do the same thing to sajax: 
- fleck
- spotter
- starry
- starrystarryprocess
- pytranspot
- SAGE
- spotrod
- spot-model
- SOAP
- ANTARESS

These codes can be broadly categorizes into three families based on their treatment of active regions: grid-based, analytical, and spherical harmonics. While the two last ones can be combined together we separate them as they are fundamentally different in their implementation. As the name makes it sound, the grid-based codes rely on tesselating the stellar surface with a grid, and parameterizing spots are regions of low flux on this grid. In contrast, the analytical approach aims to provide a closed-form solution for the transit light curve in the presence of a contaminated stellar surface. The spherical harmonics code act as a subset of this where rather than describing the active regions as hard-edged regions on the stellar surface, they instead decompose the flux of the stellar surface through spherical harmonics thus allowing for a reduction of dimensionality in that only the spherical harmonic coefficients must be provided rather than the positional arguments of each active region.

All these codes differ on their language of implementation (C, Rust, Python, Python accelerated with JAX) and in the complexity of phenomena they account for (spherical projection, differential rotation, single-band vs multi-band vs fully chromatic, distinct active region properties from the quiet photosphere, treatment of multiple active regions, dynamics of active region properties, ...). Ours adds to this list, but aims to provide implementations both at the forefront of computation time, and at modeling.

Regarding the former, sajax is implemented in JAX. This allows it to implement the diversity of modern hardware available to accelerate model evaluations, while benefiting from the interpretability of ease of access of python code. As discussed later on (or above) we also develop our code with *inference* in mind: the light curve model can be evaluated in two ways, one leveraging the just-in-time compilation advantage of JAX to speed-up the most time intensive aspect of grid-based stellar surface modeling: the building of the grid and operations done relying on this grid. The other pathway trades this JIT compilation for ease of use, combining the two function evluations into one, and providing a slick altrernative for quick and dirty model evaluation, as is needed for plotting for example (it is in fact this latter model evaluation that we use to produce the plots in most of our documentation).

# Acknowledgements
SAJAX makes use of the following dependencies: XXXXXX.
We would also like to acknowledge the base code that SAJAX was built on, SAGE, as well as ANTARESS, the code of which inspired some of the core functionalities of SAJAX.
SJM and LA would also like to thank Julien de Wit, Benjamin V. Rackham, Catriona Murray, and Zoe de Beurs for fruitful discussion that helped in the developement of SAJAX's core philosophy and functionalities.

# References
