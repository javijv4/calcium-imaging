# CALCIUM-IMAGING
Calcium imaging analyzer

# Installation
1. Create an environment and activate it. If using conda, 
```
conda create -n ca-env python=3.13.2
conda activate ca-env
```
If using .venv,
```
python -m venv .venv
source .venv/bin/activate
```
In VScode you can select the environment by using `Ctrl+Shift+P`, type Python Select Interpreter and selecting `./venv/bin/python`.

2. Install necessary packages and modules,
```
python -m pip install -e .
```

# Usage
The code has three main scripts:
1. Registration `register_images.py`
2. Preprocessing `preprocess_tissues.py`
3. Calcium Analysis `analyze_tissues.py`

**You need to run these in order**. For each one of these you will select a folder containing image stacks to be analyzed (`.mat`, `.tif`/`.tiff`, `.nd2`, or `.czi`). 

# Outputs
Below is a diagram of the inputs, codes, and outputs used in this repository.

```mermaid
graph TD;
data[sample.mat]
wdata[sample_warped.mat ]
mask[sample_tissue_mask.tif]
npz[sample_preprocessing.npz]
traces[all region traces]
regtraces["regular" traces]
tissue[tissue trace]
results[bpm, bpm_std, timing irreg., <br> upstroke time, amplitude]
results2[bpm, bpm_std, timing irreg., <br> upstroke time, amplitude]

py1("register_images.py")
py2("preprocess_tissues.py")

process1("traces with npeaks >= 2")
process21("trace analysis")
process22("trace analysis")

pngout1["sample_all_regions.png <br> sample_all_traces.png"]
pngout2["sample_region_regions.png <br> sample_region_traces.png"]

rawout[sample_raw_output.py]


data --> py1:::grey
py1:::grey --> wdata
wdata --> py2:::grey
py2:::grey --> mask
py2:::grey--> npz
mask --> traces
mask --> tissue
npz --> traces

subgraph analyze_tissues.py
    traces --> synchronicity:::blue
    tissue --> process21:::grey
    process21 --> sample_output.csv:::blue
    synchronicity --> sample_output.csv:::green
    process21 --> results:::blue
    results --> sample_output.csv
    traces --> pngout1:::green


    traces --> process1:::grey
    process1 --> regtraces
    regtraces --> process22:::grey
    process22 --> results2:::blue
    results2 --> sample_region_output.csv:::green
    regtraces --> pngout2:::green

    tissue --> rawout:::green
    regtraces --> rawout:::green
    
end


classDef grey fill:#ccc
classDef blue fill:#87ceeb
classDef green fill:#90ee90

style analyze_tissues.py fill:#f5f5f5
style analyze_tissues.py stroke:#000
```