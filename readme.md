# Overview

单细胞RNA测序（scRNA-seq）是一种高通量方法，可在单细胞水平检测基因表达，但面临dropout噪声、批次效应等挑战。本项目探讨了使用变分自编码器（VAE）处理这些问题的效果，详细研究了其原理及改进方案。通过去噪、隐空间聚类和批次校正的实验验证，改进模型性能优于原有模型，并为scRNA-seq数据分析提供了有效方法和新思路。

# Folder Contents

### datasets

数据集，内有readme.txt文件详细说明。

## PCA+tnse

文件夹为对数据集使用PCA降维，再用t-SNE可视化的代码文件，内有readme.txt文件详细说明。

## scvis实验

对数据集使用scivs模型处理，再用t-SNE可视化的代码文件，内有readme.txt文件详细说明。

## scVI实验

文件夹为对数据集使用scVI模型处理，再用t-SNE可视化的代码文件，内有readme.txt文件详细说明。

## VASC模型

文件夹为对数据集使用VASC模型处理，再用t-SNE可视化的代码文件，内有readme.txt文件详细说明。

## Denoise code


add_dropout_two&six_group.py 是对Two group数据集和Six group数据集随机加入50%噪声的代码文件。

figure1-1_retina_mean_var.py 是用RETINA数据集作基因表达方差-均值散点图的代码文件。