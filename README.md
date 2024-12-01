# GGPP

```sh
sudo mkdir -p /mnt/v1
sudo mount /dev/vdc /mnt/v1

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all

conda create -y -n py311 python=3.11
conda activate py311
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install ipython ipykernel -c pytorch -c nvidia
conda install transformers scikit-learn faiss -c pytorch -c nvidia
pip install livelossplot
conda install ipywidgets -c conda-forge  # error during download

cd /mnt/v1
git clone https://github.com/workelaina/Prompt-Perturbation-in-Retrieval-Augmented-Generation.git
mv Prompt-Perturbation-in-Retrieval-Augmented-Generation ggpp

cd ggpp
```
