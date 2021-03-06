# MMAction
this repo clone from [mmaction](https://github.com/open-mmlab/mmdetection), I had fixed it to use with custom data. I also custom [PAN](https://github.com/zhang-can/PAN-PyTorch) repo to run with custom data. 

## Introduction
MMAction is an open source toolbox for action understanding based on PyTorch.
It is a part of the [open-mmlab](https://github.com/open-mmlab) project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

### Major Features
- MMAction is capable of dealing with all of the tasks below.

  - action recognition from trimmed videos
  - temporal action detection (also known as action localization) in untrimmed videos
  - spatial-temporal action detection in untrimmed videos.


- Support for various datasets

  Video datasets have emerging throughout the recent years and have greatly fostered the devlopment of this field.
  MMAction provides tools to deal with various datasets.

- Support for multiple action understanding frameworks

  MMAction implements popular frameworks for action understanding:

  - For action recognition, various algorithms are implemented, including TSN, I3D, SlowFast, R(2+1)D, CSN.
  - For temporal action detection, we implement SSN.
  - For spatial temporal atomic action detection, a Fast-RCNN baseline is provided.

- Modular design

  The tasks in human action understanding share some common aspects such as backbones, and long-term and short-term sampling schemes.
  Also, tasks can benefit from each other. For example, a better backbone for action recognition will bring performance gain for action detection.
  Modular design enables us to view action understanding in a more integrated perspective.

## License
The project is release under the [Apache 2.0 license](https://github.com/open-mmlab/mmaction/blob/master/LICENSE).

## Updates
[OmniSource](https://arxiv.org/abs/2003.13042) Model Release (22/08/2020)
- We release several models of our work [OmniSource](https://arxiv.org/abs/2003.13042). These models are jointly trained with
Kinetics-400 and OmniSourced web dataset. Those models are of good performance (Top1 Accuracy: **75.7%** for 3-segment TSN and **80.4%** for SlowOnly on Kinetics-400 val) and the learned representation transfer well to other tasks.

v0.2.0 (15/03/2020)
- We build a diversified modelzoo for action recognition, which include popular algorithms (TSN, I3D, SlowFast, R(2+1)D, CSN). The performance is aligned with or better than the original papers.

v0.1.0 (19/06/2019)
- MMAction is online!

## Model zoo
Results and reference models are available in the [model zoo](https://github.com/open-mmlab/mmaction/blob/master/MODEL_ZOO.md).

## Installation
Please refer to [INSTALL.md](https://github.com/open-mmlab/mmaction/blob/master/INSTALL.md) for installation.

Update: for Docker installation, Please refer to [DOCKER.md](https://github.com/open-mmlab/mmaction/blob/master/DOCKER.md) for using docker for this project.

## Data preparation
If you want to prepare data with HMD51, UCF101, ... follow below links:
Please refer to [DATASET.md](https://github.com/open-mmlab/mmaction/blob/master/DATASET.md) for a general knowledge of data preparation.
Detailed documents for the supported datasets are available in `data_tools/`.
If you want to prepare custom data.
Place your video in mmaction/data/customdata/videos
with structure of directory:
```
mmaction_custom_data
├── mmaction
├── tools
├── configs
├── data
│   ├── customsdata
│   │   ├── annotations
|   |   |               ├── {your_class_name_video1}_test_split1.txt
│   │   │               |                                            ├── {video_name1} {chose(0,1,2: where 0:val, 1:train, 2:test)}
│   │   │               |                                            ├── {video_name1} {chose(0,1,2: where 0:val, 1:train, 2:test)}
│   │   │               ├── {your_class_name_video2}_test_split2.txt
│   │   │               ├── ...
│   │   │               ├── {your_class_name_videon}_test_split3.txt
│   │   ├── videos
                  ├── your_class_name_video1
                  │                         ├── video_name1
                  │                         ├── ....
                  ├──your_class_name_video2
                  │                         ├── video_name
                  │                         ├── ....     


```
1. Put your video like structure videos above
2. Run create_anotation.py --video_paths your_video_path. This will create all needed file in annotations folder for you
3. cd to mmaction_custom_data/data_tools/customdata
4. !bash extract_rgb_frames.sh
5. !bash generate_filelist.sh
remember fix your path_file in 2&3
## Get started
Please refer to [GETTING_STARTED.md](https://github.com/open-mmlab/mmaction/blob/master/GETTING_STARTED.md) for detailed examples and abstract usage.

## Contributing
We appreciate all contributions to improve MMAction.
Please refer to [CONTRUBUTING.md](https://github.com/open-mmlab/mmaction/blob/master/CONTRIBUTING.md) for the contributing guideline.

## Citation
If you use our codebase or models in your research, please cite this work.
We will release a technical report later.
```
@misc{mmaction2019,
  author =       {Yue Zhao, Yuanjun Xiong, Dahua Lin},
  title =        {MMAction},
  howpublished = {\url{https://github.com/open-mmlab/mmaction}},
  year =         {2019}
}
```

## Contact
If you have any question, please file an issue or contact the author:
```
Yue Zhao: thuzhaoyue@gmail.com
```
