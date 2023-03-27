# Cell Segmentation and Tracking using CNN-Based Distance Predictions and a Graph-Based Matching Strategy #

Segmentation and tracking method used for our [publication](#markdown-header-publication). Our submission to the 5th edition of the [ISBI Cell Tracking Challenge](http://celltrackingchallenge.net/) 2020 is based on this code (Team KIT-Sch-GE).

An improved version of the segmentation (slightly adjusted scaling & closing for the neighbor distances, training process, batch size > 1 & multi-GPU support for inference) can be found here: [https://git.scc.kit.edu/KIT-Sch-GE](https://git.scc.kit.edu/KIT-Sch-GE).

## Prerequisites
* [Anaconda Distribution](https://www.anaconda.com/products/individual)
* A CUDA capable GPU
* Minimum / recommended RAM: 16 GiB / 32 GiB
* Minimum / recommended VRAM: 12 GiB / 24 GiB

## Installation
Clone the Cell Segmentation and Tracking repository:
```
git clone https://bitbucket.org/t_scherr/cell-segmentation-and-tracking 
```
Open the Anaconda Prompt (Windows) or the Terminal (Linux), go to the Cell Segmentation and Tracking repository and create a new virtual environment:
```
cd path_to_the_cloned_repository
conda env create -f requirements.yml
```
Activate the virtual environment cell_segmentation_and_tracking_ve:
```
conda activate cell_segmentation_and_tracking_ve
```

## Segmentation
In this section, it is described how to reproduce the segmentation results of our [publication](#markdown-header-publication). Download the Cell Tracking Challenge training data sets [BF-C2DL-HSC](http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-HSC.zip), [BF-C2DL-MuSC](http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-MuSC.zip), [Fluo-N2DL-HeLa](http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip), and [Fluo-N3DH-CE](http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-CE.zip). Unzip the data sets into the folder *datasets*. Download the [evaluation software](http://public.celltrackingchallenge.net/software/EvaluationSoftware.zip) from the Cell Tracking Challenge and unzip it in the repository. To train on your own data, our implemented Pytorch Dataset can be used but the training data need to be of similar shape. In addition, the inference function needs to be adapted.

### CTC Training Set
After downloading the required Cell Tracking Challenge data, the CTC Training Set can be created with:
```
python cell_segmentation.py --create_ctc_set
```

### Train Segmentation Models
The models specified in the file *segmentation_settings.json* can be trained on the CTC Training Set with:
```
python cell_segmentation.py --train
```

### Evaluate Models
Trained Models can be evaluated on the test sets with:
```
python cell_segmentation.py --evaluate
```

### Plot Results
Boxplots and exemplary crops of the best models of each method (for each cell type) can be created with:
```
python cell_segmentation.py --plot
```
Note: the number of iterations/initializations/models of each method needs to be odd.

## Tracking
This section describes how to reproduce the tracking results of our [publication](#markdown-header-publication).

It is assumed the data set follows the same folder structure and file naming as the CTC data sets:
```
dataset
└───01
└───01_GT
└───01_RES
└───02
└───02_GT
└───02_RES
```
First, run a segmentation approach to derive segmentation masks for dataset/0x, where x is either 1 or 2.
The tracking with the same parametrization as in the paper can be derived by executing:
```
python run_tracking.py img_path segm_path res_path
```
where img_path is the path to the folder dataset/0x and segm_path the path to the folder containing the segmentation masks. The resulting tracking masks and lineage file will be stored in res_path.
If segm_path and res_path are the same path, the segmentation masks will be replaced by the tracking masks.




## Publication ##
T. Scherr, K. Löffler, M. Böhland, and R. Mikut (2020). Cell Segmentation and Tracking using CNN-Based Distance Predictions and a Graph-Based Matching Strategy. PLoS ONE 15(12). DOI: [10.1371/journal.pone.0243219](https://doi.org/10.1371/journal.pone.0243219).

## License ##
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.