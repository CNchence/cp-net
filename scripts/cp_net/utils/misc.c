#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// #include <vector>
// #include <algorithm>


double mean1d_up_limit(double* x, int len_x, double uplim)
{
  int i;
  double ret = 0.0;
  for(i = 0 ; i < len_x ; i++){
    ret += fmin(x[i], uplim);
  }
  ret /= len_x;
  return ret;
}

double right_tmean1d_up_limit(double* x, int len_x, double thre, double uplim)
{
  int i;
  double cnt = 0.0;
  double ret = 0.0;
  for(i = 0 ; i < len_x ; i++){
    if (x[i] > thre){
      continue;
    }
    ret += fmin(x[i], uplim);
    cnt++;
  }
  ret /= cnt;
  return ret;
}

int compare(const void *p, const void *q)
{
    if( *(double*)p > *(double*)q ) return 1;
    if( *(double*)p < *(double*)q ) return -1;
    return 0;
}


double right_percentile1d(double* x, int len_x, int thre)
{
  int outlier_num = len_x * thre / 100.0;
  double* x_sort = x;
  qsort(x_sort, len_x, sizeof(double), compare);
  return x_sort[outlier_num];
}


double visibility_scoring(double* x, int len_x, int percentile_thre, double max_dist)
{
  double pthre = right_percentile1d(x, len_x, percentile_thre);
  return right_tmean1d_up_limit(x, len_x, pthre, max_dist);
}


double calc_visib_socre_from_map(double* depth_diff, double* mask, int im_h, int im_w,
                                 int visib_thre, double percentile_thre, double max_dist_lim)
{
  int w, h, p, nonzero_cnt = 0;
  int i, inlier_num;
  double val, ret = 0.0;
  double* score_arr;
  score_arr = (double *)malloc(sizeof(double) * im_h * im_w);

  // extract nonzero
  for(h = 0; h < im_h; h++){
    for(w = 0; w < im_w; w++){
      p = h * im_w + w;
      val = fabs(depth_diff[p] * mask[p]);
      if (val != 0){
        score_arr[nonzero_cnt++] = fabs(val);
      }
    }
  }
  if(nonzero_cnt < visib_thre){return 1e15;}
  score_arr = (double *)realloc(score_arr, sizeof(double) * nonzero_cnt);

  // percentile
  inlier_num = nonzero_cnt * percentile_thre / 100.0;
  qsort(score_arr, nonzero_cnt, sizeof(double), compare);
  // score sum
  for(i = 0 ; i < inlier_num ; i++){
    ret += fmin(score_arr[i], max_dist_lim);
  }
  ret /= inlier_num;

  return ret;
}


double calc_invisib_socre_from_map(double* depth_diff, double* mask, int im_h, int im_w,
                                   double fore_thre, double percentile_thre, double max_dist_lim)
{
  int w, h, p, nonzero_cnt = 0;
  int i, inlier_num;
  double val, ret = 0.0;
  double* score_arr;
  score_arr = (double *)malloc(sizeof(double) * im_h * im_w);

  // extract nonzero
  for(h = 0; h < im_h; h++){
    for(w = 0; w < im_w; w++){
      p = h * im_w + w;
      val = depth_diff[p] * mask[p];
      if(val != 0){
        if (val > fore_thre){
          val = 0;
        }
        score_arr[nonzero_cnt++] = fabs(val);
      }
    }
  }
  if(nonzero_cnt == 0){return 0;}
  score_arr = (double *)realloc(score_arr, sizeof(double) * nonzero_cnt);

  // percentile
  inlier_num = nonzero_cnt * percentile_thre / 100.0;
  qsort(score_arr, nonzero_cnt, sizeof(double), compare);
  // score sum
  for(i = 0 ; i < inlier_num ; i++){
    ret += fmin(score_arr[i], max_dist_lim);
  }
  ret /= inlier_num;

  return ret;
}

void pointcloud_to_depth_impl(double* pc, double* K, double* depth,
                              int im_h, int im_w, int len_pc)
{
  int i, xx, yy;
  double zz, val = 0;
  for(i = 0; i < len_pc; i++){
    zz = pc[2 * len_pc + i];
    xx = pc[i] * K[0] / zz + K[2];
    yy = pc[len_pc + i] * K[4] / zz + K[5];
    if(xx >= 0 && xx < im_w && yy >= 0 && yy < im_h){
      val = depth[im_w * yy + xx];
      if (val == 0){
        depth[im_w * yy + xx] = zz;
      }
      else{
        depth[im_w * yy + xx] = fmin(zz, val);
      }
    }
  }
}
