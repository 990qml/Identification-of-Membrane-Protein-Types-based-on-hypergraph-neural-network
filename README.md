1、Environment configuration requirements: Install (Pytorch 0.4.0) (https://pytorch.org/). Also need to install yaml.

2、Datasets: Processed datasets are in the data folder and the membrane protein feature representation algorithm is in the feature folder. Data processing and configuration are described in the datasets folder

3、Run: Configuration file:config.yaml,D4 is the target dataset and can be replaced train.py: select the data to be trained, and then start training. See the code comments for the specific process, save the prediction results to A.csv of the corresponding dataset in data/protein
