# TransGCN

TransGCN: a semi-supervised graph convolution network-based framework to infer protein translocations in spatio-temporal proteomics



## Install dependencies



- First clone the repository.

```shell
git clone https://github.com/wangbing587/TransGCN.git
```



- It's recommended to create a separate conda environment for running TransGCN

```shell
#create an environment called TransGCN
conda create -n TransGCN python=3.8

#activate your environment
conda activate TransGCN
```



- The  package is developed based on the Python libraries [torch](https://pytorch.org/get-started/previous-versions/) and [torch-geometric](https://pypi.org/project/torch-geometric/) (*PyTorch Geometric*) framework, and can be run on GPU (recommend) or CPU.

(i)  torch==1.13.1  (CPU) 

```shell
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

(ii) torch==1.13.1+cu116  (GPU)

```shell
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```



- Install other the required packages.

```shell
pip install -r requirement.txt
```



- Required package list

```shell
copy
warnings
argparse==1.1
numpy==1.19.2
pandas==1.1.3
scikit-learn==0.23.2
joblib==0.17.0
torch_geometric==2.3.1
torch==1.13.1  (CPU) or torch==1.13.1+cu116  (GPU)
```



## Data  format

- the data format must be .csv (comma-separated values).
- the first column must be ProteinID.
-  the last column must be markers, which record the protein subcellular localization (PSL). For proteins with unknown PSLs, they should be marked as 'unknown'.
- the middle rows must be  the fraction expression of protiens.
- For a detailed format, refer to  **valerio2022.csv**, which five paired replicates.



##  Activate TransGCN environment

Before running TransGCN, ensure that the dependent packages are installed and the TransGCN environment is activated as follows:

```
# activate TransGCN environment
conda activate TransGCN
```



## TransGCN help

the main.py in TrainGCN file is a python script for convenience using TransGCN by Command node

The introduction of theTransGCN parameters can be achieved by naming them as follows

```shell
cd TrainGCN
python main.py -h
```

Output

```shell
python main.py -h
usage: main.py [-h] [-f F] [-r R]

TransGCN: a semi-supervised graph convolution network-based framework to infer protein translocations in spatio-
temporal proteomics

optional arguments:
  -h, --help      show this help message and exit
  -f F, --file F  file path, first column must be ProteinID, last column must be markers
  -r R, --rep R   number of repeated paird experiments
```





## Run TransGCN

the main.py can be run from the command line interface with the following commands, where -f (dataset file) and -r (the number of parid replicates) are the two required parameters. An example command is “python main.py -f valerio2022.csv -r 5”, in which “-f” is input dataset file and “-r” is the number of replicates. 

```shell
python main.py -f valerio2022.csv -r 5
```

### Reference

Bing Wang, Xiangzheng Zhang, Xudong Han, Bingjie Hao, Yan Li, Xuejiang Guo, TransGCN: a semi-supervised graph convolution network–based framework to infer protein translocations in spatio-temporal proteomics, *Briefings in Bioinformatics*, Volume 25, Issue 2, March 2024, bbae055, https://doi.org/10.1093/bib/bbae055



If there are any problems, please contact me.

Bing Wang, E-mail: wangbing587@163.com
