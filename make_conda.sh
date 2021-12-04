
env=$1

source ~/miniconda3/etc/profile.d/conda.sh
conda create -n $env python=3.8
conda activate $env

PYTHONPATH=$(python -c 'import site; print(site.getsitepackages()[0])')
echo "PATH : ${PYTHONPATH}"
python setup.py develop

