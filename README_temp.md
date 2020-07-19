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

    cd ./code/
    python visualize.py ./X.image.png ./X.region.png

and you will see the visualization like this:

![img1](https://lllyasviel.github.io/DanbooRegion/page_imgs/result_sim.png)

## Something about the dataset quality

In fact we have an user study to visualize the quality of this dataset.

![img1](https://lllyasviel.github.io/DanbooRegion/page_imgs/us.jpg)

*The x-axis is the quantity of regions in each region map, while the y-axis is the user scoring with the above scoring standard.*

We can see that **about 50% annotations only have normal (or below) quality** and **about 30% annotations are even unfinished**. Besides, **70% annotations still need corrections**. Only **about 20% annotations look good or beautiful**. This is because annotating these regions is very difficult and time-consuming. Thanks a lot to all 12 artists involved in this data annotating projects! The involved artists are:

![img1](https://lllyasviel.github.io/DanbooRegion/page_imgs/human.jpg)

*Some artists do not want to be named and got mosaics.*

Although the quality of this dataset is not perfect, we can still do lots of things that are very useful for cartoon image processing applications.

**We are continuously improving the quality of this dataset. If you check back in a few months, the quality of the dataset may have improved many times.**

# 2. Converting regions to learnable skeleton maps.

Firstly, let's say you have an image like this:

<img src="https://lllyasviel.github.io/DanbooRegion/page_imgs/sk/input.jpg" width = "300" />

Then, you have asked some users or artists to annotate the image and get the region map like this:

<img src="https://lllyasviel.github.io/DanbooRegion/page_imgs/sk/region.png" width = "300" />

The problem is that how we can train neural networks to predict the region maps? It is obvious that we cannot train a pix2pix to predict the region map because the colors in the region maps are random, and L1 loss is meaningless. Therefore, we will need some additional steps.

We first detect the edge of all regions using OpenCV and get this edge map:

<img src="https://lllyasviel.github.io/DanbooRegion/page_imgs/sk/edge.png" width = "300" />

After that, we use the **skeletonize** function to extract the region skeletons. The function can be found at

    from skimage.morphology import thin as skeletonize

and the result is like this:

<img src="https://lllyasviel.github.io/DanbooRegion/page_imgs/sk/skeleton.png" width = "300" />

After that, we randomly initialize an image with skeletons and edges like this:

<img src="https://lllyasviel.github.io/DanbooRegion/page_imgs/sk/field.jpg" width = "300" />

Then, we smooth the above image to get the final skeleton map like this:

<img src="https://lllyasviel.github.io/DanbooRegion/page_imgs/sk/height.jpg" width = "300" />

Note that this skeleton map is learnable and you can use anything to learn it. For example, you can train pix2pix or pix2pixHD to predict the skeleton map.

Also, the skeleton map can be translated to normal maps like this:

<img src="https://lllyasviel.github.io/DanbooRegion/page_imgs/sk/normal.jpg" width = "300" />

Or you can thresold the skeleton map to get some watershed markers like this:

<img src="https://lllyasviel.github.io/DanbooRegion/page_imgs/sk/mark.png" width = "300" />

Note that we provide full codes for you to do all the above things! Just run

    cd ./code/
    python skeletonize.py ./region_test.png

Have fun with the codes.

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

