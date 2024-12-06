# Saturation and Shale Volume Analysis Tool using Ploty and Dash

## Overview
This project provides Python functions to analyze well log data for calculating saturations and shale volumes, as well as generating visualizations to interpret subsurface properties. The tools leverage **Plotly** and **Dash**for interactive plotting and are designed to work with data that includes depth, resistivity, porosity, and other well log parameters.

---

## Features

### 1. Saturation Plotting (`Plot_sw2`)
**Purpose:**  
Calculate and visualize bulk volumes of water and hydrocarbons, saturation profiles, and Pickett plots.

**Inputs:**  
- `m`, `n`: Archie's parameters.  
- `rw`: Formation water resistivity.  
- `mslope`: Slope for trends in apparent porosity.  
- `depth_range`: Depth range for analysis (tuple in the format `(upper, lower)`).

**Outputs:**  
- **Bulk Volume Plot:** Visualizes bulk volume water (BVW) and bulk volume oil (BVO) with depth.  
- **Pickett Plot:** A log-log plot of porosity and resistivity for saturation interpretation.  
- **Vsh vs. Mstar Apparent Plot:** Crossplot showing trends in shale content and apparent Archie’s parameter.

---

### 2. Shale Volume Plotting (`shale_plot3`)
**Purpose:**  
Visualize gamma ray, spontaneous potential (SP) logs, and calculate shale volume using multiple methods.

**Inputs:**  
- `sp_clean`, `sp_shale`: SP clean and shale values.  
- `gr_clean`, `gr_shale`: GR clean and shale values.  
- `neut_shale`: Neutron porosity for shale.  
- `dt_matrix`, `dt_shale`: Sonic transit time for matrix and shale.  
- `depth_range`: Depth range for analysis.  

**Outputs:**  
- **SP and GR Logs:** SP and GR curves against depth, with clean and shale reference lines.  
- **Histograms:** SP and GR histograms to assess data distribution.  
- **Crossplots:** Neutron-density and NPHI-RHOB crossplots.  
- **Shale Volumes:** Calculations using GR, SP, DT, and other methods.

---
## Demo  
Below is a demonstration of the Saturation and Shale Volume Analysis Tool in action:  

![Saturation and Shale Volume Analysis Tool Demo](demo(2).gif)  




## Dependencies
- Python 3.10+  
- Libraries:  
  - `numpy`  
  - `pandas`  
  - `plotly`  
  - `dash`  
  - `scipy`

Install dependencies using:

```bash
pip install numpy pandas plotly dash scipy
