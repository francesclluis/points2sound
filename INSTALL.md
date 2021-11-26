This project has been tested with Ubuntu 18.04 using the following main libraries: `pytorch=1.4` and `MinkowskiEngine=0.4.1`. 
You will need a full installation of CUDA 10.1 in order to compile MinkowskiEngine v0.4.1

```
# create a conda environment and install requirements for MinkowskiEngine v0.4.1

conda create -n Points2Sound python=3.7
conda activate Points2Sound
conda install numpy openblas
conda install pytorch torchvision -c pytorch

# Install MinkowskiEngine v0.4.1

wget https://github.com/NVIDIA/MinkowskiEngine/archive/refs/tags/v0.4.1.tar.gz
tar -zxvf v0.4.1.tar.gz
cd MinkowskiEngine-0.4.1/
python setup.py install

# install additional libraries

conda install scipy
conda install -c conda-forge librosa
pip install open3d
conda install -c anaconda scikit-image

```
With this newly created environment, you can start using the repo.
