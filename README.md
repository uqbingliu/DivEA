# DivEA

This repo is for reproducing our work *High-quality Task Division for Large-scale Entity Alignment*, which has been accepted at CIKM 2022
([arXiv](High-quality Task Division for Large-scale Entity Alignment)).

Download the code and [data](https://www.dropbox.com/sh/1ecy19x7j6f4bak/AAA4rY28rizHe1AFlNVlYEAqa?dl=0). The structure of folders should be organized as below
```text
divea/
|- datasets/    # datasets are put under this folder
   |- dbp15k/
   |- dwy100k/
   |- 2m/   # dataset fb_dbp of size 2M
|- divea/   # code of our method
|- RREA/    # RREA model
|- GCN-Align/     # GCN-Align model
|- scripts/    # scripts files for running our method with RREA
|- scripts2/    # scripts files for running our method with GCN-Align
|- environment.yml   # conda environment file
|- README.md
```


## Python Environment

`cd` to project directory firstly. 

Create the environment named divea and install most packages by running command:
```shell
conda env create -f environment.yml
```

Then, activate the environment:
```shell
conda activate divea
```

Finally, install package `networkx-metis` as below. Other installation instructions of `networkx-metis` can be found [here](https://networkx-metis.readthedocs.io/en/latest/install.html).
```shell
git clone https://github.com/networkx/networkx-metis.git
cd networkx-metis/
python setup.py build
python setup.py install
```


## Run scripts

The scripts for running our method with RREA are put under `scripts/`.
* `bash run_over_perf_vs_cps.sh`. Overall performance. Table 1.
* `bash run_over_perf_vs_sbp.sh`. Overall performance. Table 2.
* `bash run_over_perf_vs_cps_2m.sh`. Overall performance. Table 1.
* `bash run_over_perf_vs_sbp_2m.sh`. Overall performance. Table 2.

The scripts for running our method with GCN-Align are put under `scripts2/`. The script file names and corresponding functions can be aligned with scripts under `scripts/`.


## Citation
tab

## Acknowledgement

We used the source codes of [RREA](https://github.com/MaoXinn/RREA) and [GCN-Align](https://github.com/1049451037/GCN-Align).





