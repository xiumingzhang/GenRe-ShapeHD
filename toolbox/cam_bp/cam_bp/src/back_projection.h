int back_projection_forward(THCudaTensor* depth, THCudaTensor* camdist, THCudaTensor* fl, THCudaTensor* voxel, THCudaTensor* cnt);
int back_projection_backward(THCudaTensor* depth, THCudaTensor* fl, THCudaTensor* camdist, THCudaTensor* cnt, THCudaTensor* grad_in, THCudaTensor* grad_depth, THCudaTensor* grad_camdist, THCudaTensor* grad_fl);
int get_surface_mask(THCudaTensor* depth, THCudaTensor* camdist, THCudaTensor* fl, THCudaTensor* cnt, THCudaTensor* mask);
int spherical_back_proj_forward(THCudaTensor* depth, THCudaTensor* grid_in, THCudaTensor* voxel, THCudaTensor* cnt);
int spherical_back_proj_backward(THCudaTensor* depth, THCudaTensor* grid_in, THCudaTensor* cnt, THCudaTensor* grad_in, THCudaTensor* grad_depth);
