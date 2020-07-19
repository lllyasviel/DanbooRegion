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

Using the above transform, you can translate the full training data into skeleton maps. Just run

    cd ./code/
    python skeletonize_all.py

This will automatically generate all training region skeletons, and the logs are like this:

    Writing   ./DanbooRegion2020/train/0.skeleton.png   1/3377
    Writing   ./DanbooRegion2020/train/1.skeleton.png   2/3377
    Writing   ./DanbooRegion2020/train/2.skeleton.png   3/3377
    Writing   ./DanbooRegion2020/train/3.skeleton.png   4/3377
    Writing   ./DanbooRegion2020/train/4.skeleton.png   5/3377
    Writing   ./DanbooRegion2020/train/5.skeleton.png   6/3377
    ...
    Writing   ./DanbooRegion2020/train/3376.skeleton.png   3377/3377

Before running this script, the file structures are

    ../code/DanbooRegion2020
    ../code/DanbooRegion2020/train
    ../code/DanbooRegion2020/train/X.image.png
    ../code/DanbooRegion2020/train/X.region.png

After this script, the file structures become

    ../code/DanbooRegion2020
    ../code/DanbooRegion2020/train
    ../code/DanbooRegion2020/train/X.image.png
    ../code/DanbooRegion2020/train/X.region.png
    ../code/DanbooRegion2020/train/X.skeleton.png

And this is a screenshot after all skeletons are generated:

![img1](https://lllyasviel.github.io/DanbooRegion/page_imgs/sc2.jpg)

Note that the selected files in the above screenshot are the generated skeleton maps. Now you have prepared everything.

# 4. Training a neural network to predict the skeleton maps.

Before training the model, make sure that you have followed the instractions in *3. Converting the entire dataset to learnable skeleton maps*.

We provide full codes for you to train a model on this dataset:

    cd ./code/
    python train.py

We note that:

* If you run this script in the first time, it will generate a new model and train it directly.
* During training, the model will be saved in each 200 iterations.
* If you run this script when a saved model is avaliable, it will read the previous model and continue training.
* If you do not want to continue the training, please delete the "saved_model.net" and "saved_discriminator.net" file.
* This code will train a U-net with adversarial and perceptural L1 loss, which has similar performance with pix2pixHD.
* The script uses many data augmentation methods. Please see the code for details.
* The training is not very expensive. You may even train it on a laptop with GTX980M.

During the training, the script will save the neural network estimated skeleton maps for each 10 iterations. The saved estimations are in the path:

    ../code/results
    ../code/results/0.jpg
    ../code/results/1.jpg
    ../code/results/2.jpg
    ...

The entire training will take 50000 iterations.

On my very old laptop (GTX980M+I7CPU), the successful logs are

    Using TensorFlow backend.
    2020-07-19 20:44:39.820506: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
    2020-07-19 20:44:40.423216: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1030] Found device 0 with properties: 
    name: GeForce GTX 980M major: 5 minor: 2 memoryClockRate(GHz): 1.1265
    pciBusID: 0000:01:00.0
    totalMemory: 8.00GiB freeMemory: 6.71GiB
    2020-07-19 20:44:40.423930: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 980M, pci bus id: 0000:01:00.0, compute capability: 5.2)
    saved_model.net--created
    saved_discriminator.net--created
    0/50000 Currently 13.778525352478027 second for each iteration.
    1/50000 Currently 2.0006484985351562 second for each iteration.
    2/50000 Currently 1.6455974578857422 second for each iteration.
    3/50000 Currently 1.8700015544891357 second for each iteration.
    4/50000 Currently 1.8167574405670166 second for each iteration.
    5/50000 Currently 1.5877196788787842 second for each iteration.
    6/50000 Currently 1.6236474514007568 second for each iteration.
    7/50000 Currently 1.7862193584442139 second for each iteration.

If you have better GPU than me, your speed should be much faster.

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

