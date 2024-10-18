# 文件夹内容介绍

model_config.yaml 文件为scvis的模型参数文件，运行py代码前需调节好参数。


scvis_two.py 将dropout数据用scvis模型处理后生成的重构基因表达值，在py代码文件中需要手动修改重构基因表达值为Gaussian、student、NB或ZINB。

figure4-1&4-2_heatmap_two.py	文件为使用 x_hat_two 前缀的csv文件作热图的代码.

figure4-3&4-4_heatmap_six.py	文件为使用 x_hat_six 前缀的csv文件作热图的代码.

figure4-5d-g_scvis_cortex.py	文件为scvis处理CORTEX数据集的代码文件.

figure4-6d-g_scvis_pbmc68k.py	文件为scvis处理PBMC68k数据集的代码文件.

figure4-7d-g_scvis_mouse.py	文件为scvis处理mouse cell atlas数据集的代码文件.

figure4-8d-g_scvis_RETINA.py	文件为scvis处理RETINA数据集的代码文件.

figure4-9d-g_scvis_simulation.py 文件为scvis处理Simulation数据集的代码文件.

## heatmap_result

文件夹内为论文去噪部分最终挑选的20×20的基因表达矩阵，储存为csv格式。