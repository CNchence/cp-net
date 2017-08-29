#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>

using namespace Eigen;

void ransac_estimation_loop(double* x_arr, double* y_arr,
                            double* depth, double* model, double* K, double* obj_mask,
                            int len_arr, int len_model, int im_h, int im_w, int n_ransac,
                            double max_thre, int pthre, double* ret_t, double* ret_r)
{
  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_int_distribution<> rand_sample(0, len_arr);

  int i, j, k, r_sample;
  double score, best_score = 1e15;

  MatrixXd x_arr_mat = Map<Matrix<double, Dynamic, Dynamic> >(x_arr, 3, len_arr);
  MatrixXd y_arr_mat = Map<Matrix<double, Dynamic, Dynamic> >(y_arr, 3, len_arr);
  MatrixXd depth_mat = Map<Matrix<double, Dynamic, Dynamic> >(depth, im_h, im_w);

  MatrixXd x_demean(3, len_arr);
  MatrixXd y_demean(3, len_arr);
  // todo
  // double* -> eigen

  MatrixXd x_mat(3,3);
  MatrixXd y_mat(3,3);
  VectorXd x_mean(3);
  VectorXd y_mean(3);

  MatrixXd R(3,3);
  MatrixXd best_R(3,3);
  VectorXd t(3);
  VectorXd best_t(3);

  // for svd
  MatrixXd v, u;
  VectorXd dist(len_arr);

  // for pointcloud -> depth
  MatrixXd model_mat = Map<Matrix<double, Dynamic, Dynamic> >(model, 3, len_model);
  MatrixXd model_depth = MatrixXd::Zero(im_h, im_w);
  MatrixXd depth_diff;

  VectorXd xs(len_model);
  VectorXd ys(len_model);
  VectorXd zs =  model_mat.row(2);;
  int xx, yy;
  double val;

  VectorXd score_visib_arr;
  VectorXd score_invisib_arr;

  for(i = 0; i < n_ransac; i++){
    // random sampling
    for (j = 0; j < 3; j++){
      r_sample = rand_sample(mt);
      x_mat.col(j) = x_arr_mat.col(r_sample);
      y_mat.col(j) = y_arr_mat.col(r_sample);
    }
    x_mean = x_mat.rowwise().mean();
    y_mean = y_mat.rowwise().mean();
    x_demean = x_arr_mat.colwise() - x_mean;
    y_demean = y_arr_mat.colwise() - y_mean;

    // compute SVD
    JacobiSVD<MatrixXd> svd(x_demean * y_demean.transpose(), ComputeFullU | ComputeFullV);
    v = svd.matrixV();
    u = svd.matrixU();
    // Compute R = V * U'
    if ((u * v).determinant() < 0){
      for (int x = 0; x < 3; ++x)
        v (x, 2) *= -1;
    }
    R = v * u.transpose();
    t = y_mean - R * x_mean;
    dist = (((R * x_arr_mat).colwise() + t) - y_arr_mat).cwiseAbs().colwise().sum();
    score = dist.sum();

    // pointcloud -> depth image
    // std::cout << zz.inverse().rows() << std::endl;
    // std::cout << zz.inverse().cols() << std::endl;
    xs = model_mat.row(0) * K[0]  * zs  + K[2] * VectorXd::Ones(len_model);
    ys = model_mat.row(1) * K[4]  * zs  + K[5] * VectorXd::Ones(len_model);

    for(j = 0; j < len_model; j++){
      xx = xs(j);
      yy = ys(j);
      if(xx >= 0 && xx < im_w && yy >= 0 && yy < im_h){
        val = model_depth(yy, xx);
        if (val == 0){
          model_depth(yy, xx) = zs(j);
        }
        else{
          model_depth(yy, xx) = fmin(xs(j), val);
        }
      }
    }

    depth_diff = model_depth - depth_mat;

    // todo
    // awesome code !!
    //

    if(score < best_score){
      best_t = t;
      best_R = R;
    }
  }

  // // Eigen -> double*
  for(j = 0 ; j < 3 ; j++){
    ret_t[j] = best_t(j);
    for(k = 0 ; k < 3 ; k++){
      ret_r[j * 3 + k] = best_R(j, k);
    }
  }
}

void calc_rot_eigen_svd3x3(double* y_arr, double* x_arr, double* out_arr)
{
  MatrixXd x_mat(3,3);
  MatrixXd y_mat(3,3);
  int i;
  for(i = 0; i < 9; i++){
    x_mat(i) = x_arr[i];
    y_mat(i) = y_arr[i];
  }
  // compute svd
  JacobiSVD<MatrixXd> svd(x_mat * y_mat.transpose(), ComputeFullU | ComputeFullV);
  MatrixXd v = svd.matrixV();
  MatrixXd u = svd.matrixU();
  // Compute R = V * U'
  if ((u * v).determinant() < 0){
    for (int x = 0; x < 3; ++x)
      v (x, 2) *= -1;
  }
  MatrixXd out_mat = v * u.transpose();
  for(i = 0; i < 9; i++){
    out_arr[i] = out_mat(i);
  }
}

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
