source activate shaperecon
conda config --add channels conda-forge
conda install shapely rtree pyembree
conda install -c conda-forge scikit-image
conda install "pillow<7"
pip install trimesh[all]==2.35.47
