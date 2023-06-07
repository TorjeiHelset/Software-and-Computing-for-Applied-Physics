# Monte Carlo Simulation of X-ray imaging

X-rays are electromagnetical waves with energies in the region 0.1 keV to 100 keV. The energy of X-ray beams will generally lie between 10 keV and 100 keV in medical imaging. Different materials have different dampening coefficients, which means that x-rays will move through these material to a varying degree. TBecause of this varying degree of dampening, it is possible to get a contrast between different materials when the waves are detected after moving through the material. In particular, bones and tissue have different dampening coefficients, which is why x-rays, among other things, are used to look for fractures in bones.

Even though x-ray imaging is a very useful medical tool, the interaction between the radiation and the matter it moves through can be dangerous, especially in higher dosages. Therefore, it is desirable to minimize the radiation, while still maintaining a high enough contrast so that any damage is easily visible.

To this end, it is interesting to simulate photon propagation through different materials to get an idea of how much radiation is needed to get a good contrast. This project will simulate this photon propagation using a Monte Carlo method.
