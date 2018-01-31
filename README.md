# cp-net
object pose estimation using neural network


## package explanation
### cp_net
pose estimation using deep neural network

<<<<<<< HEAD
=======
### cp_net_ros
ros wrapper of cp_net


>>>>>>> 0e0225c710a86a8a892d78d02fff915d006ecb39
## Data Set
linemod dataset


## requirements
<<<<<<< HEAD
Eigen, cython, chainer v3, cuda, scikit-learn, cv2, etc...
=======
Eigen, cython, chainer v3, cuda, etc...
>>>>>>> 0e0225c710a86a8a892d78d02fff915d006ecb39

https://github.com/oshiroy/pose_estimation_gpu


## Sample
```
pip install -e .
mkdir train_data
cd train_data
## download train_data from (https://drive.google.com/file/d/1A2skh4mYgsoXXKnLOuQ2BVPsnSoe8PZu/view?usp=sharing) in train_data
tar xfvz linemodSIXD2017.tar.gz
cd ..

mkdir trained_data
cd trained_data
## download network model from (https://drive.google.com/file/d/1fiBsCbx0kWK76lhZZ-EGDrRJ35hFBNsu/view?usp=sharing) in trained_data
tar xfvz model_iteration-100000.tar.gz
cd ..

# demo
<<<<<<< HEAD
cd examples/linemodSIXD2017
cd python demo.py -g 0 ../../trained_data/model_iteration-100000
=======
cd examples/linemodSIXD
cd python demo.py -g 0 ../../model_iteration-100000
>>>>>>> 0e0225c710a86a8a892d78d02fff915d006ecb39

```
