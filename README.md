# GGPP

```sh
conda create -y -n py311 python=3.11
conda activate py311
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install ipython ipykernel -c pytorch -c nvidia
conda install transformers scikit-learn faiss -c pytorch -c nvidia
pip install livelossplot
conda install ipywidgets -c conda-forge  # error during download
# conda install ipywidgets -c pytorch

git clone https://github.com/workelaina/Prompt-Perturbation-in-Retrieval-Augmented-Generation.git
mv Prompt-Perturbation-in-Retrieval-Augmented-Generation ggpp
cd ggpp
```
