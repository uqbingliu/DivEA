# DivEA

This repo is for reproducing our work *High-quality Task Division for Large-scale Entity Alignment*, which has been accepted at CIKM 2022
([arXiv](https://arxiv.org/pdf/2208.10366.pdf)).

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

Please cite this paper if you use the released code in your work.
```
@inproceedings{DBLP:conf/cikm/LiuHZZZ22,
  author    = {Bing Liu and
               Wen Hua and
               Guido Zuccon and
               Genghong Zhao and
               Xia Zhang},
  editor    = {Mohammad Al Hasan and
               Li Xiong},
  title     = {High-quality Task Division for Large-scale Entity Alignment},
  booktitle = {Proceedings of the 31st {ACM} International Conference on Information
               {\&} Knowledge Management, Atlanta, GA, USA, October 17-21, 2022},
  pages     = {1258--1268},
  publisher = {{ACM}},
  year      = {2022},
  url       = {https://doi.org/10.1145/3511808.3557352},
  doi       = {10.1145/3511808.3557352},
  timestamp = {Wed, 04 Jan 2023 07:33:22 +0100},
  biburl    = {https://dblp.org/rec/conf/cikm/LiuHZZZ22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Acknowledgement

We used the source codes of [RREA](https://github.com/MaoXinn/RREA) and [GCN-Align](https://github.com/1049451037/GCN-Align).





