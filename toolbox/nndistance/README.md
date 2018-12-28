# Chamfer Distance for Pytorch 
Modified from [pointGAN](https://github.com/fxia22/pointGAN)

## Requirements
Tested on Pytorch 0.3.1
Due to syntax change in Pytorch 0.4.0, this implementation probably won't work on Pytorch 0.4.0

## Install 
```bash
./clean.sh
./setup.sh script
```
Note that currently the code only supports building as script, so you'll need to put this directory under your code's root directory, where you can import using `import nndistance`

## Example
Run `test.py` as an example:

```bash
cp test.py ..
python test.py
```

## Usage
- The function `nndistance.functions.nndistance(pts1, pts2)` return two lists of distances - the closest distance for each point in `pts1` to point cloud `pts2`, and the closest distance for each point in `pts2` to point cloud `pts1`. 
  - For convenience, the distance here is defined as `(x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2)`, **without taking the square root**. 
  - If you want to take the square root, keep in mind that in Pytorch, **the gradient of `sqrt(0)` is `nan`**, so you'll probably want to add a small `eps` before taking sqrt. 
- The function `nndistance.functions.nndistance_score(pts1, pts2)` return a list of scores. 


Internal note: this implementation gives the same result as our previously used implementation in tensorflow.  