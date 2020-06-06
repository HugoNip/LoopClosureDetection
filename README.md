## Introduction
This project shows loop closure detection in SLAM, including:  
1. Create vocabulary for datasets with both small (10 images) and big (1000+ images) size.
2. Compute similarity score between two images, which can be used for loop closure detection.

## Requirements

### DBow3 Package
#### Source
https://github.com/rmsalinas/DBow3

#### Compile and Install
```
cd [path-to-Eigen]
mkdir build
cd build
cmake ..
make 
sudo make install 
```

### OpenCV
#### Required Packages
OpenCV  
OpenCV Contrib

gcc version: gcc (Ubuntu 5.4.0-6ubuntu1/~16.04.12) 5.4.0 20160609   
g++ version: g++ (Ubuntu 5.4.0-6ubuntu1/~16.04.12) 5.4.0 20160609  
(Note: OpenCV will fail to compile with gcc/g++ of 9.2.0 version)

## Compile this Project
```
mkdir build
cd build
cmake ..
make 
```

## Run
```
./feature_training  
./gen_vocab_large  
./loop_closure  
```


## Reference
[Source](https://github.com/HugoNip/slambook2/tree/master/ch11)
