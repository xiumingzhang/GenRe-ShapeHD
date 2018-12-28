
#ifdef __cplusplus
extern "C" {
#endif

  int back_projection_forward_wrap (THCState* state, THCudaTensor* depth, THCudaTensor* camdist, THCudaTensor* fl, THCudaTensor* voxel, THCudaTensor* cnt);
  int back_projection_backward_wrap (THCState* state, THCudaTensor* depth, THCudaTensor* fl, THCudaTensor* camdist, THCudaTensor* cnt, THCudaTensor* grad_in, THCudaTensor* grad_depth, THCudaTensor* grad_camdist, THCudaTensor* grad_fl);
  int get_surface_mask_wrap(THCState* state, THCudaTensor* depth, THCudaTensor* camdist, THCudaTensor* fl, THCudaTensor* cnt, THCudaTensor* mask);
  int spherical_back_proj_forward_wrap(THCState* state, THCudaTensor* depth, THCudaTensor* grid_in, THCudaTensor* voxel, THCudaTensor* cnt);
  int spherical_back_proj_backward_wrap(THCState* state, THCudaTensor* depth, THCudaTensor* grid_in, THCudaTensor* cnt, THCudaTensor* grad_in, THCudaTensor* grad_depth);
#ifdef __cplusplus
}
#endif
