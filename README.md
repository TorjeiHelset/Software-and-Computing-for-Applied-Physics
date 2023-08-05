# Monte Carlo Simulation of X-ray imaging
## Table on contents 

- [Introduction](#introduction)
- [Installing Project](#installing-project)
- [Interacting with the Project](#ibteracting-with-the-project)

<!----><a name="introduction"></a>
### Introduction
X-rays are electromagnetical waves with energies in the region 0.1 keV to 100 keV. The energy of X-ray beams will generally lie between 10 keV and 100 keV in medical imaging. Different materials have different dampening coefficients, which means that x-rays will move through these material to a varying degree. TBecause of this varying degree of dampening, it is possible to get a contrast between different materials when the waves are detected after moving through the material. In particular, bones and tissue have different dampening coefficients, which is why x-rays, among other things, are used to look for fractures in bones.

Even though x-ray imaging is a very useful medical tool, the interaction between the radiation and the matter it moves through can be dangerous, especially in higher dosages. Therefore, it is desirable to minimize the radiation, while still maintaining a high enough contrast so that any damage is easily visible.

To this end, it is interesting to simulate photon propagation through different materials to get an idea of how much radiation is needed to get a good contrast. This project will simulate this photon propagation using a Monte Carlo method.

<!----><a name="installing-projectn"></a>
### Installing Project
To install the project on your computer first make sure that you have python and pip installed. Then enter in the terminal the following lines
```bash
cd enter/your/location/of/choice
```
```bash
git clone https://github.com/TorjeiHelset/Software-and-Computing-for-Applied-Physics.git
```
To install the necessary libraries enter the following command after navigating to the cloned repository.
```bash
pip install requirements.txt
```

### Interacting with the project
The project can be interacted with in two main ways. The simplest way is to run the notebook and change parameters as desired.
The second way is to run the Streamlit application. This can be done by first running the following command:

```bash
streamlit run xray_app.py
```

By running this command you will start a local server. You can now open the URL that appears in the terminal in a browser. This will open the Streamlit application, and you can run the simulation, change parameters and display plots as you wish.
