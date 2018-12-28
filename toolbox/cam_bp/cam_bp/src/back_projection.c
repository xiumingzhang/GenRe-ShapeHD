#include <stdbool.h>
#include <stdio.h>
#include <THC/THC.h>
#include "back_projection.h"
#include "back_projection_kernel.h"

extern THCState *state;

int back_projection_forward(THCudaTensor* depth, THCudaTensor* camdist, THCudaTensor* fl, THCudaTensor* voxel, THCudaTensor* cnt){
  int success = 0;
  success = back_projection_forward_wrap(state, depth, camdist, fl, voxel, cnt);
  // check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}


int back_projection_backward(THCudaTensor* depth, THCudaTensor* fl, THCudaTensor* camdist, THCudaTensor* cnt, THCudaTensor* grad_in, THCudaTensor* grad_depth, THCudaTensor* grad_camdist, THCudaTensor* grad_fl){
  int success = 0;
  success = back_projection_backward_wrap(state, depth, fl, camdist, cnt, grad_in, grad_depth, grad_camdist, grad_fl);
  // check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}

int get_surface_mask(THCudaTensor* depth, THCudaTensor* camdist, THCudaTensor* fl, THCudaTensor* cnt, THCudaTensor* mask){
  int success = 0;
  success = get_surface_mask_wrap(state, depth, camdist, fl, cnt, mask);
  // check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}

int spherical_back_proj_forward(THCudaTensor* depth, THCudaTensor* grid_in, THCudaTensor* voxel, THCudaTensor* cnt){
  int success = 0;
  success = spherical_back_proj_forward_wrap(state,  depth, grid_in, voxel, cnt);
  // check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}
int spherical_back_proj_backward(THCudaTensor* depth, THCudaTensor* grid_in, THCudaTensor* cnt, THCudaTensor* grad_in, THCudaTensor* grad_depth){
  int success = 0;
  success = spherical_back_proj_backward_wrap(state,  depth, grid_in,cnt,grad_in,grad_depth);
  // check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}
