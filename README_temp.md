# The DanbooRegion 2020 Dataset

![img0](https://lllyasviel.github.io/DanbooRegion/page_imgs/teaser.jpg)

DanbooRegion is a project conducted by ToS2P (the Team of Style2Paints), aiming at find a solution to extract regions from illustrations and cartoon images, so that many region-based image processing algrithoms can be applied to in-the-wild illustration and digital paintings. The main uniqueness of this project is that our dataset is created by real artists in a semi-automatic manner. 

This project begins at 2019 Jan and a techinical paper, namely "DanbooRegion: An Illustration Region Dataset", is accepted to the European Conference on Computer Vision (ECCV) 2020.

![img1](https://lllyasviel.github.io/DanbooRegion/page_imgs/ex.jpg)

*- Some example illustration and the corresponding region annotations.*

[Project Page] [Paper] [Zhihu]

# Table of Contents

In this page, we provide the following contents:

    0. Preparation.
    1. Downloding the region annotation dataset.
    2. Converting regions to learnable skeleton maps.
    3. Converting the entire dataset to learnable skeleton maps.
    4. Training a neural network to predict the skeleton maps.
    5. Converting a skeleton map to regions.
    6. Developing your own framework and train it on this dataset.
    7. Benchmarking the performace of your model.

However, if you are busy dating your girlfriend, or in a great hurry to write a paper, you can skip the 01234567 above and directly jump to the 89 below.

    8. Downloding the pre-trained model.
    9. Testing the model with an image.

We also provide some additional playful contents.

    10. Playing with some applications.
    11. Playing with some configuations.

# 0. Preparation.

Whether you want to download the dataset or the pretrained model, you will need a **Python 3.6** environment with appropriate **CUDA** installed, and then clone the repo:

    git clone https://github.com/lllyasviel/DanbooRegion.git

and then install the pip environments via

    pip install tensorflow_gpu==1.5.0
    pip install keras==2.2.4
    pip install opencv-python==3.4.2
    pip install numpy==1.15.4
    pip install numba==0.39.0
    pip install scipy==1.1.0
    pip install scikit-image==0.13.0
    pip install scikit-learn==0.22.2

Other versions may also works but the above versions are tested by myself.

# 1. Downloding the region annotation dataset.

You can download this dataset from

    Google Drive:
    An URL will be bere.

or if you do not have access to google

    Baidu Drive (百度网盘):
    An URL will be bere.

After the downloading and decompression, you may see images like this:

![img1](https://lllyasviel.github.io/DanbooRegion/page_imgs/sc.jpg)

The file structure is

    ../code/DanbooRegion2020
    ../code/DanbooRegion2020/DanbooRegion2020UNet.net
    ../code/DanbooRegion2020/train
    ../code/DanbooRegion2020/train/X.image.png
    ../code/DanbooRegion2020/train/X.region.png
    ../code/DanbooRegion2020/val
    ../code/DanbooRegion2020/val/X.image.png
    ../code/DanbooRegion2020/val/X.region.png
    ../code/DanbooRegion2020/test
    ../code/DanbooRegion2020/test/X.image.png

Below is an example of "X.image.png"

![img1](https://lllyasviel.github.io/DanbooRegion/page_imgs/image.jpg)

and the example of "X.region.png".

![img1](https://lllyasviel.github.io/DanbooRegion/page_imgs/regions_sim.png)

We also provide a script for you to visualize the regions by filling the original colors to all regions, you can use the script by:

    python visualize.py ./X.image.png ./X.region.png

and you will see the visualization like this:

![img1](https://lllyasviel.github.io/DanbooRegion/page_imgs/result_sim.png)

## Something about the dataset quality

balabala.

# 2. Converting regions to learnable skeleton maps.



# 3. Converting the entire dataset to learnable skeleton maps.



# 4. Training a neural network to predict the skeleton maps.



# 5. Converting a skeleton map to regions.



# 6. Developing your own framework and train it on this dataset.



# 7. Benchmarking the performace of your model.



# 8. Downloding the pre-trained model.



# 9. Testing the model with an image.



# 10. Playing with some applications.



# 11. Playing with some configuations.


# Citation

If you use this code for your research, please cite our paper:

    Bib file coming soon.

# 中文社区

我们有一个除了技术什么东西都聊的以技术交流为主的宇宙超一流二次元相关技术交流吹水群“纸片协会”。如果你一次加群失败，可以多次尝试。

    纸片协会总舵：184467946

