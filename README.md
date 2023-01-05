# No-Reset Object Detection with Spiking Neural Networks

*This work is supported by the French technological research agency (ANRT) through a CIFRE thesis in collaboration between Renault and Université Côte d'Azur.*

This repository contains the codes for the Chapter 5 entitled *Designing High-Performance SNNs* of my PhD thesis [Performance of spiking neural networks on event data for embedded automotive applications](https://theses.fr/s313551).

We achieved new state-of-the-art results for SNNs on event data, both on classification (Prophesee N-CARS) and object detection (Prophesee GEN1) tasks.

Our main contributions are:
1. How to fuse Batch Normalization and Convolution layers when the order BN-CONV is used. (see `fuse_other_modules.py`)
2. Two new SNNs backbones, entitled **ST-VGG** and **ResCat-SNN**, using a *patchify* stem, our BN-CONV order, PLIF neurons and concatenation-based residual connections. (see the `models` module)
3. Two new SNNs for object detection, ST-VGG+SSD and ResCat-SNN+SSD. (see the `models` module)
4. A new training method for SNNs on object detection based on Truncated BPTT that leverages the temporal continuity of event data during the training, removing the need to reset the membrane potentials between each sample. (see `object_detection_module.py`)

# Results

## Object Detection on Prophesee GEN1

| **Models** | **#Params** | **ACCs/ts** | **COCO mAP &#8593;** | **Sparsity &#8595;** |
|---|:---:|:---:|:---:|:---:|
| 16-ST-VGG+SSD | 642k    | 125M  | 0.135 |  37.66\% |
| 32-ST-VGG+SSD | 1.14M   | 453M  | 0.184 |  36.24\% |
| 64-ST-VGG+SSD | 2.88M   | 1.73G | 0.203 |  38.87\% |
| ResCat-SNN-16+SSD | 759k    | 165M  | 0.147 |  33.71\% |
| ResCat-SNN-32+SSD | 1.43M   | 596M  | 0.192 |  35.59\% |
| ResCat-SNN-64+SSD | 4.57M   | 2.27G | 0.203 |  45.14\% |


Train a 32-ST-VGG+SSD model on Prophesee GEN1:

    python object_detection.py -path path/to/GEN1_dataset -backbone vgg -save_ckpt

To measure test mAP and sparsity on a pretrained model:

    python object_detection.py -path path/to/GEN1_dataset -pretrained path/to/pretrained_model -no_train -test

Other parameters are available in `object_detection.py`.

## Classification on Prophesee NCARS and Prophesee GEN1 Classification datasets

| **Models** | **#Params** | **ACCs/ts** | **NCARS acc &#8593;** | **NCARS sparsity &#8595;** | **GEN1 Classif acc &#8593;** | **GEN1 Classif sparsity &#8595;** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 16-ST-VGG     | 125k   | 6.01M   | 0.917          | 23.36\% | 0.897          | 24.30\% |
| 32-ST-VGG     | 499k   | 23.27M  | 0.941          | 22.64\% | 0.920          | 29.03\% |
| 64-ST-VGG     | 1.994M | 93.06M  | 0.925          | 41.61\% | 0.906          | 34.85\% |
| ResCat-SNN-16 | 216k   | 7.78M   | 0.927          | 28.73\% | 0.893          | 30.18\% |
| ResCat-SNN-32 | 862k   | 30.59M  | 0.935          | 26.62\% | 0.917          | 33.25\% |
| ResCat-SNN-64 | 3.442M | 121.31M | 0.940          | 36.05\% | 0.901          | 41.33\% |

# Citation

If you find this work useful feel free to cite our PhD thesis:

    Loïc Cordone, "Performance of spiking neural networks on event data for embedded automotive applications", PhD Thesis, Université Côte d'Azur, Dec. 2022.

<br>

    @phdthesis{cordone_performance_2022,
	    type = {These de doctorat},
	    title = {Performance of spiking neural networks on event data for embedded automotive applications},
	    url = {https://www.theses.fr/s313551},
	    urldate = {2023-01-04},
	    school = {Université Côte d'Azur},
	    author = {Cordone, Loïc},
	    collaborator = {Miramond, Benoît},
	    month = dec,
	    year = {2022},
    }


