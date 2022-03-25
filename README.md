Semi-supervised Cross-scale Graph Prototypical Network for Hyperspectral Image Classification, TNNLS, 2022.
==
 [Bobo Xi](https://scholar.google.com/citations?user=O4O-s4AAAAAJ&hl=zh-CN), [Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), [Yunsong Li](https://dblp.uni-trier.de/pid/87/5840.html),  [Rui song](https://scholar.google.com/citations?user=_SKooBYAAAAJ&hl=zh-CN), [Yuchao Xiao](https://ieeexplore.ieee.org/author/37089176254), [Qian Du](https://my.ece.msstate.edu/faculty/du/) and [Jocelyn Chanussot](https://jocelyn-chanussot.net/).
***
Code for paper: [Semi-supervised Cross-scale Graph Prototypical Network for Hyperspectral Image Classification.](https://ieeexplore.ieee.org/document/9740412) 

<div align=center><img src="/Image/frameworks.jpg" width="80%" height="80%"></div>
Fig. 1: Structure diagram of the proposed X-GPN for HSIC. It comprises four components: multiscale adjacency matrices construction, cross-scale feature learning, SBAA, and a novel prototypical layer.

Training and Test Process
--
Please run the 'main_IP.py' to reproduce the X-GPN results on [IndianPines](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Indian_Pines) data set. The training samples distribution and the obtained classification map are shown below. We have successfully test it on Ubuntu 16.04 with Tensorflow 1.13.1 and Keras 2.1.5. 

<div align=center><p float="center">
<img src="/Image/false_color.jpg" height="150"/>
<img src="/Image/gt.jpg" height="150"/>
<img src="/Image/training_map.jpg" height="150"/>
<img src="/Image/classification_map.jpg" height="150"/>
</p></div>
<div align=center>Fig. 2: The composite false-color image, groundtruth, training samples, and classification map of Indian Pines dataset.</div>  

 Visualization of the feature distribution by t-SNE
 --
<div align=center><p float="center">
<img src="/Image/softmax.jpg" height="200"/>
<img src="/Image/dce.jpg" height="200"/>
<img src="/Image/dce_ter.jpg" height="200"/>
</p></div>
<div align=center>Fig. 2: Visualization of the feature distribution obtained by CE, DCE, DCE + TER loss functions on Indian Pines dataset.</div>

References
--
If you find this code helpful, please kindly cite:

[1] B. Xi, J. Li, Y. Li, R. Song, Y. Xiao, Q. Du, J. Chanussot, “Semi-supervised Cross-scale Graph Prototypical Network for Hyperspectral Image Classification,” IEEE Transactions on Neural Networks and Learning Systems, pp. 1-15, 2022, [doi:10.1109/TNNLS.2022.3158280](https://ieeexplore.ieee.org/document/9740412). 

[2] Y. Li, B. Xi, J. Li, R. Song, Y. Xiao and J. Chanussot, "SGML: A Symmetric Graph Metric Learning Framework for Efficient Hyperspectral Image Classification," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 15, pp. 609-622, 2022, [doi: 10.1109/JSTARS.2021.3135548](https://ieeexplore.ieee.org/abstract/document/9652087).

[3] B. Xi, J. Li, Y. Li and Q. Du, "Semi-Supervised Graph Prototypical Networks for Hyperspectral Image Classification," 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS, 2021, pp. 2851-2854, [doi: 10.1109/IGARSS47720.2021.9553372](https://ieeexplore.ieee.org/document/9553372).

Citation Details
--
BibTeX entry:
```
@ARTICLE{Xi_2022TNNLS_XGPN,
  author={Xi, Bobo and Li, Jiaojiao and Li, Yunsong and Song, Rui and Xiao, Yuchao and Du, Qian and Chanussot, Jocelyn},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Semi-supervised Cross-scale Graph Prototypical Network for Hyperspectral Image Classification}, 
  year={2022},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TNNLS.2022.3158280}}
```
```
@ARTICLE{Xi_2021JSTARS_SGML,
  author={Li, Yunsong and Xi, Bobo and Li, Jiaojiao and Song, Rui and Xiao, Yuchao and Chanussot, Jocelyn},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={SGML: A Symmetric Graph Metric Learning Framework for Efficient Hyperspectral Image Classification}, 
  year={2022},
  volume={15},
  number={},
  pages={609-622},
  doi={10.1109/JSTARS.2021.3135548}}
```
```
@INPROCEEDINGS{Xi_2021IGARSS_GPN,
  author={Xi, Bobo and Li, Jiaojiao and Li, Yunsong and Du, Qian},
  booktitle={2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS}, 
  title={Semi-Supervised Graph Prototypical Networks for Hyperspectral Image Classification}, 
  year={2021},
  volume={},
  number={},
  pages={2851-2854},
  doi={10.1109/IGARSS47720.2021.9553372}}
```
 
Licensing
--
Copyright (C) 2022 Bobo Xi

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
