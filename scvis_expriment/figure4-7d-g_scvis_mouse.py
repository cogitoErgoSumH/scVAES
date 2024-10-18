#!/usr/bin/env python
# coding: utf-8

## 参数
# =============================================================================
# hyperparameter: {
#   optimization: {
#     method: Adam,
#     learning_rate: 0.01 ## default = 0.01
#   },
# 
#   batch_size: 512,  ## default = 512
#   max_epoch: 300,
#   regularizer_l2: 0.001,
# 
#   perplexity: 10,  ## default = 10
# 
#   seed: 1
# }
# 
# architecture: {
#   latent_dimension: 8, default = 2
# 
#   inference: {
#     layer_size: [128, 64, 32],  ## default = [128, 64, 32]
#   },
# 
#   model: {
#     layer_size: [32, 64, 128], ## default = [32,32,32,64,128]
#   },
# 
#   activation: "ELU"  ## default = ELU
# }
# =============================================================================


# In[1]:

### tf_helper.py


import warnings
warnings.filterwarnings("ignore")

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # 这两句是恢复使用tensorflow 1.0的语法
tf.logging.set_verbosity(tf.logging.ERROR)

# Glorot 正态分布初始化器，也称为 Xavier 正态分布初始化器。
# 它从以 0 为中心，标准差为 stddev = sqrt(2 / (fan_in + fan_out)) 的截断正态分布中抽取样本，
# 其中 fan_in 是权值张量中的输入单元的数量， fan_out 是权值张量中的输出单元的数量。
# 这个初始化器是用来使得每一层输出的方差应该尽量相等，使用正态分布来随机初始化，返回值是一个初始化器
X_INIT = tf.keras.initializers.glorot_normal()


# tf.Variable创建变量，提供的初始值是 X_INIT(shape=fan_in_out)，
# trainable=True 开启梯度，后面可以进行微分； 变量命名为 name='weight'，
# 但这个函数后面没用到。
def xavier(fan_in_out, name='weight'):
    return tf.Variable(X_INIT(fan_in_out), trainable=True, name=name)


# tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换，这里转换为float32，
# 函数返回值是初始化好的权重，后面会用到。
def weight_xavier_relu(fan_in_out, name='weight'):
    stddev = tf.cast(tf.sqrt(2.0 / fan_in_out[0]), tf.float32)
    initial_w = tf.truncated_normal(shape=fan_in_out,
                                    mean=0.0, stddev=stddev)

    return tf.Variable(initial_w, trainable=True, name=name)


# 这个函数后面没用到，像是选这三个函数其中一个来用。
def weight_variable(fan_in_out, name='weight'):
    initial = tf.truncated_normal(shape=fan_in_out,
                                  mean=0.0, stddev=0.1)
    return tf.Variable(initial, trainable=True, name=name)


# 类似的，这个函数是对偏置进行初始化
def bias_variable(fan_in_out, mean=0.1, name='bias'):
    initial = tf.constant(mean, shape=fan_in_out)
    return tf.Variable(initial, trainable=True, name=name)


# 返回的是一个元组，表示变量的shape，这个函数后面用到很多次
def shape(tensor):
    return tensor.get_shape().as_list()


# In[2]:

### tsne_helper.py

import sys
import numpy as np

MAX_VAL = np.log(sys.float_info.max) / 2.0 # 获取Python中最大的浮点数，约为1.8E+308，取对数再除以2
# MAX_VAL = 354.891...

np.random.seed(0) # 设置随机种子


# 计算过渡概率，x是传入的数据，困惑度perplexity设置为30
# 这个函数没看懂，猜测是计算tsne定义距离的公式，文中有给出
def compute_transition_probability(x, perplexity=30.0,
                                   tol=1e-4, max_iter=50, verbose=False):
    # x should be properly scaled so the distances are not either too small or too large
    # x应该适当缩放，这样距离既不会太小也不会太大
    
    if verbose:
        print('tSNE: searching for sigma ...')

    (n, d) = x.shape
    #print("in compute_transition_probability, n = ", n)
    #print("in compute_transition_probability, d = ", d)
    sum_x = np.sum(np.square(x), 1) # 逐个对元素取平方后，将每一行的元素相加，sum_x的shape=(1,n)

    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x) ## n × n 的对称矩阵，对角线元素为0， 位置ij上的元素是xi到xj的二范数距离
    p = np.zeros((n, n)) # n × n， 元素全为0

    # Parameterized by precision
    beta = np.ones((n, 1)) # n × 1 矩阵 元素全为1
    entropy = np.log(perplexity) / np.log(2)  # perplexity(pi) = 2^H(pi) = 2^(-SUM(p · log2 p))

    # Binary search for sigma_i ## 二分法查找sigma_i
    idx = range(n)
    for i in range(n):
        idx_i = list(idx[:i]) + list(idx[i+1:n]) # 跳过i， 对于每一个 idx_i = [0,1,...i-1, i+1, ...,  n] 共 n-1 个元素

        beta_min = -np.inf
        beta_max = np.inf

        # Remove d_ii
        dist_i = dist[i, idx_i] # 元素i与其他n-1个元素的距离列表，长度为n-1
        h_i, p_i = compute_entropy(dist_i, beta[i])
        h_diff = h_i - entropy

        iter_i = 0
        while np.abs(h_diff) > tol and iter_i < max_iter:
            if h_diff > 0:
                beta_min = beta[i].copy()
                if np.isfinite(beta_max):
                    beta[i] = (beta[i] + beta_max) / 2.0
                else:
                    beta[i] *= 2.0
            else:
                beta_max = beta[i].copy()
                if np.isfinite(beta_min):
                    beta[i] = (beta[i] + beta_min) / 2.0
                else:
                    beta[i] /= 2.0

            h_i, p_i = compute_entropy(dist_i, beta[i])
            h_diff = h_i - entropy

            iter_i += 1

        p[i, idx_i] = p_i

    if verbose:
        print('Min of sigma square: {}'.format(np.min(1 / beta)))
        print('Max of sigma square: {}'.format(np.max(1 / beta)))
        print('Mean of sigma square: {}'.format(np.mean(1 / beta)))

    return p


def compute_entropy(dist=np.array([]), beta=1.0):
    p = -dist * beta
    shift = MAX_VAL - max(p)
    p = np.exp(p + shift)
    sum_p = np.sum(p)

    h = np.log(sum_p) - shift + beta * np.sum(np.multiply(dist, p)) / sum_p

    return h, p / sum_p


# In[3]:

# likelihood.py

import numpy as np

EPS = 1e-20


# 对数正态分布的概率密度表达式，后面没用到这个
def log_likelihood_gaussian(x, mu, sigma_square):
    return tf.reduce_sum(-0.5 * tf.log(2.0 * np.pi) - 0.5 * tf.log(sigma_square) -
                         (x - mu) ** 2 / (2.0 * sigma_square), 1)

# 对数t分布的概率密度表达式，后面用到了这个，应该是二选一来用
def log_likelihood_student(x, mu, sigma_square, df=2.0):
    sigma = tf.sqrt(sigma_square)
    dist = tf.distributions.StudentT(df=df, loc=mu, scale=sigma)
    print(tf.reduce_sum(dist.log_prob(x), reduction_indices=1),
          type(tf.reduce_sum(dist.log_prob(x), reduction_indices=1))
          )
    return tf.reduce_sum(dist.log_prob(x), reduction_indices=1)

## 从scVI复制过来的
## 根据NB模型，minibatch的对数似然（标量）
def log_nb_positive(x, mu, theta, eps=1e-8):
    """
    log likelihood (scalar) of a minibatch according to a nb model. 
    
    Variables:
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    eps: numerical stability constant
    """
    (n, d) = x.shape
    nn = x.get_shape().as_list()[0]
    dd = x.get_shape().as_list()[1]
    print("in log_nb_positive, type(x) = ", type(x))
    print("in log_nb_positive, shape(x) = ", x.shape)
    print("in log_nb_positive, type(n) = ", type(n))
    print("in log_nb_positive, type(n) = ", type(d))
    print("in log_nb_positive, n = ", n)
    print("in log_nb_positive, d = ", d)
    print("in log_nb_positive, nn = ", nn)
    print("in log_nb_positive, dd = ", dd)
    res = tf.lgamma(x + theta) - tf.lgamma(theta) - tf.lgamma(x + 1) + x * tf.log(mu + eps) \
                                - x * tf.log(theta + mu + eps) + theta * tf.log(theta + eps) \
                                - theta * tf.log(theta + mu + eps)
    return tf.reduce_sum(res, reduction_indices=1)

def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    """
    log likelihood (scalar) of a minibatch according to a zinb model. 

    Notes:
    We parametrize the bernouilli using the logits, hence the softplus functions appearing
    # 我们使用logit参数化伯努利函数，因此出现了softplus函数
    
    Variables:
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)  # 负二项分布的均值（需大于0），shape = minibatch × genes
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes) # 逆扩散系数（需大于0），shape = minibatch × genes
    pi: logit of the dropout parameter (real support) (shape: minibatch x genes) # dropout参数的logit（实数），shape = minibatch × genes
    eps: numerical stability constant # 数值稳定常数
    """
    (n, d) = x.shape
    nn = x.get_shape().as_list()[0]
    dd = x.get_shape().as_list()[1]
    print("in log_zinb_positive, type(x) = ", type(x))
    print("in log_zinb_positive, shape(x) = ", x.shape)
    print("in log_zinb_positive, type(n) = ", type(n))
    print("in log_zinb_positive, type(n) = ", type(d))
    print("in log_zinb_positive, n = ", n)
    print("in log_zinb_positive, d = ", d)
    print("in log_zinb_positive, nn = ", nn)
    print("in log_zinb_positive, dd = ", dd)
    ## 见论文Supple Note3
    case_zero = tf.nn.softplus(- pi + theta * tf.math.log(theta + eps) - theta * tf.math.log(theta + mu + eps)) \
                                - tf.nn.softplus( - pi)
    # case_zero = log[(θ/(θ+π))^θ]
    case_non_zero = - pi - tf.nn.softplus(- pi) \
                                + theta * tf.math.log(theta + eps) - theta * tf.math.log(theta + mu + eps) \
                                + x * tf.math.log(mu + eps) - x * tf.math.log(theta + mu + eps) \
                                + tf.math.lgamma(x + theta) - tf.math.lgamma(theta) - tf.math.lgamma(x + 1)
    # case_non_zero = logNB(x; mu, theta)
    
    mask = tf.cast(tf.less(x, eps), tf.float32)  # 若x的某个元素为0，该位置为1，否则为0。tf.less返回两个张量各元素比较（x<y）得到的真假值组成的张量; tf.cast()数据类型转换
    res = tf.multiply(mask, case_zero) + tf.multiply(1 - mask, case_non_zero)
    # 综合0和非0两种情况，得到最终的似然表达式。tf.multiply()相同位置的元素相乘，mask是ZINB的参数π，也是神经网络fh训练出来的参数
    return tf.reduce_sum(res, reduction_indices=1)


# In[4]:

### vae.py

from collections import namedtuple

#LAYER_SIZE = [128, 64, 32]
#OUTPUT_DIM = 2
KEEP_PROB = 1.0
EPS = 1e-6
MAX_SIGMA_SQUARE = 1e10
MAX_THETA = 1e10

# 定义一个namedtuple类型LocationScale，并包含mu和sigma_square属性
LocationScale_encoder = namedtuple('LocationScale_encoder', ['mu', 'sigma_square'])
LocationScale_decoder = namedtuple('LocationScale_decoder', ['mu', 'sigma_square','theta','px_dropout'])


# =============================================================================
# MLP, Muti-Layer Perception 多层感知机
class MLP(object):
    def __init__(self, input_data, input_size, layer_size, output_dim,
                 activate_op=tf.nn.elu, # exponential linear unit，elu结合了sigmoid和Relu
                 init_w_op=weight_xavier_relu, # 前面 tf_helper.py 定义的初始化函数，用于初始化权重
                 init_b_op=bias_variable): # 前面 tf_helper.py 定义的初始化函数，用于初始化偏置
        self.input_data = input_data # 输入数据
        self.input_dim = shape(input_data)[1] # 输入维度为输入数据的列数，即特征数
        self.input_size = input_size # 输入数据的数量，即样本数量

        self.layer_size = layer_size # 隐层每一层的单元数，这里在前面设定为[128, 64, 32]
        self.output_dim = output_dim # 输出维度，前面设定为2

        self.activate, self.init_w, self.init_b = activate_op, init_w_op, init_b_op # 激活函数、初始权重、初始偏置

        # tf.name_scope()、tf.variable_scope()会在模型中开辟各自的空间，而其中的变量均在这个空间内进行管理
        # 这里是关于编码器网络的设定
        with tf.name_scope('encoder-net'):
            # 储存input_dim → 128的参数
            self.weights = [self.init_w([self.input_dim, layer_size[0]])] # 相当于 
            self.biases = [self.init_b([layer_size[0]])] # 一共有128个偏置参数，相当于阈值，偏置个数由下一层的单元数决定
            
            # wx+b 完成 input_dim → 128 的计算过程
            self.hidden_layer_out = tf.matmul(self.input_data, self.weights[-1]) + self.biases[-1]
            self.hidden_layer_out = self.activate(self.hidden_layer_out) # 激活函数处理
            
            # 循环遍历
            for in_dim, out_dim in zip(layer_size, layer_size[1:]):
                # 储存 128 → 64 → 32 的参数
                self.weights.append(self.init_w([in_dim, out_dim])) # 把每层之间的权重储存起来
                self.biases.append(self.init_b([out_dim])) # 把每层的偏置储存起来
                # 每层输出都用激活函数处理，完成 128 → 64 → 32 的计算过程
                self.hidden_layer_out = self.activate(
                    tf.matmul(self.hidden_layer_out, self.weights[-1]) + self.biases[-1]
                    ) 

# 高斯VAE，继承了上面的MLP类
class GaussianVAE(MLP):
    def __init__(self, input_data, input_size,
                 layer_size,#=LAYER_SIZE, # LAYER_SIZE = [128, 64, 32]
                 output_dim,#=OUTPUT_DIM, # OUTPUT_DIM = 默认 2
                 decoder_layer_size): # [32, 64 ,128]
        super(self.__class__, self).__init__(input_data, input_size,
                                             layer_size, output_dim) # super用于继承

        self.num_encoder_layer = len(self.layer_size) # =3 encoder部分一共3层网络
        
        # 这里是关于mu的隐藏层设定，储存 32 → 2 的参数
        with tf.name_scope('encoder-mu'):
            self.bias_mu = self.init_b([self.output_dim]) # 一共2个偏置
            self.weights_mu = self.init_w([self.layer_size[-1], self.output_dim])
        
        # 这里是关于sigma的隐藏层设定，储存 32 → 2 的参数
        with tf.name_scope('encoder-sigma'):
            self.bias_sigma_square = self.init_b([self.output_dim]) # 这两句跟上面一样，只是命名不一样
            self.weights_sigma_square = self.init_w([self.layer_size[-1], self.output_dim])
        
        with tf.name_scope('encoder-parameter'):
            self.encoder_parameter = self.encoder() # encoder函数在下面，返回值是 mu 和 sigma_square 两个参数
        
        # 重参数化技巧对隐空间的z采样
        with tf.name_scope('sample'): 
            self.ep = tf.random_normal(
                [self.input_size, self.output_dim],
                mean=0, stddev=1, name='epsilon_univariate_norm')

            self.z = tf.add(self.encoder_parameter.mu,
                            tf.sqrt(self.encoder_parameter.sigma_square) * self.ep,
                            name='latent_z')

        self.decoder_layer_size = decoder_layer_size # [32, 64, 128]
        self.num_decoder_layer = len(self.decoder_layer_size) # 3
        
        # 这里是关于解码器网络的设定
        # 储存 2 → 32 的参数
        with tf.name_scope('decoder'):
            self.weights.append(self.init_w([self.output_dim, self.decoder_layer_size[0]]))
            self.biases.append(self.init_b([self.decoder_layer_size[0]])) # 一共32个偏置参数
            
            # wx+b 并用激活函数处理，到这里输出为32维
            self.decoder_hidden_layer_out = self.activate(
                tf.matmul(self.z, self.weights[-1]) +
                self.biases[-1]) 
            
            # 循环遍历
            for in_dim, out_dim in zip(self.decoder_layer_size, self.decoder_layer_size[1:]):
                # 储存 32 → 64 → 128 的参数
                self.weights.append(self.init_w([in_dim, out_dim])) # 跟上面类似
                self.biases.append(self.init_b([out_dim]))
                
                # 类似， wx+b ，并用激活函数处理，循环结束后，完成 32 → 64 → 128 的计算过程
                self.decoder_hidden_layer_out = self.activate(
                    tf.matmul(self.decoder_hidden_layer_out, self.weights[-1]) +
                    self.biases[-1]) #
            
            # 下面这三一样的，命名不同，储存 128 → input_dim 的参数mu、sigma和px_dropout
            self.decoder_bias_mu = self.init_b([self.input_dim]) # 一共 input_dim 个偏置
            self.decoder_weights_mu = self.init_w([self.decoder_layer_size[-1], self.input_dim])

            self.decoder_bias_sigma_square = self.init_b([self.input_dim])
            self.decoder_weights_sigma_square = self.init_w([self.decoder_layer_size[-1], self.input_dim])

            self.decoder_bias_theta = self.init_b([self.input_dim])
            self.decoder_weights_theta = self.init_w([self.decoder_layer_size[-1], self.input_dim])
            
            self.decoder_bias_px_dropout = self.init_b([self.input_dim])
            self.decoder_weights_px_dropout = self.init_w([self.decoder_layer_size[-1], self.input_dim])
            
            # 下面这两句也是一样的，命名不同， wx+b 并加激活函数 完成 128 → input_dim 的计算过程
            mu = tf.add(tf.matmul(self.decoder_hidden_layer_out,
                                  self.decoder_weights_mu),
                        self.decoder_bias_mu)
            mu_nb = tf.nn.softplus(mu)
            sigma_square = tf.add(tf.matmul(self.decoder_hidden_layer_out,
                                            self.decoder_weights_sigma_square),
                                  self.decoder_bias_sigma_square)
            sigma_square = tf.clip_by_value(tf.nn.softplus(sigma_square),EPS, MAX_SIGMA_SQUARE)
            theta = tf.Variable(tf.random_normal([self.input_dim]))
            theta = tf.clip_by_value(tf.nn.softplus(theta),EPS, MAX_THETA)
            px_dropout = tf.add(tf.matmul(self.decoder_hidden_layer_out,
                                            self.decoder_weights_px_dropout),
                                  self.decoder_bias_px_dropout) 
            # tf.clip_by_value(A, min, max) 让张量A中 小于min的元素等于min，大于max的元素等于max
            # 把最终的参数 mu 和 sigma_square 赋值给 decoder_parameter
            if args.recon_distribution == 'gaussian':
                self.decoder_parameter = LocationScale_decoder(mu,
                                                               sigma_square,
                                                               None,
                                                               None)
            elif args.recon_distribution == 'student':
                self.decoder_parameter = LocationScale_decoder(mu,
                                                               sigma_square,
                                                               None,
                                                               None)
            elif args.recon_distribution == 'nb':
                self.decoder_parameter = LocationScale_decoder(mu_nb,
                                                               None,
                                                               theta,
                                                               None)
            elif args.recon_distribution == 'zinb':
                self.decoder_parameter = LocationScale_decoder(mu_nb,
                                                               None,
                                                               theta,
                                                               px_dropout)
    # 传入隐变量z，完成整个解码器的过程，最后输出 mu 和 sigma_square 两个参数
    def decoder(self, z):
        # wx+b 并用激活函数处理
        hidden_layer_out = self.activate(
            tf.matmul(z, self.weights[self.num_encoder_layer]) +
            self.biases[self.num_encoder_layer]
        )

        for layer in range(self.num_encoder_layer+1,
                           self.num_encoder_layer + self.num_decoder_layer):
            hidden_layer_out = self.activate(
                tf.matmul(hidden_layer_out, self.weights[layer]) +
                self.biases[layer])

# =============================================================================
#         mu = tf.add(tf.matmul(hidden_layer_out, self.decoder_weights_mu),
#                     self.decoder_bias_mu)
#         sigma_square = tf.add(tf.matmul(hidden_layer_out,
#                                         self.decoder_weights_sigma_square),
#                               self.decoder_bias_sigma_square)
#         param = LocationScale_decoder(mu, tf.clip_by_value(tf.nn.softplus(sigma_square),
#                                                     EPS, MAX_SIGMA_SQUARE))        
# =============================================================================

        mu = tf.add(tf.matmul(hidden_layer_out,
                              self.decoder_weights_mu),
                    self.decoder_bias_mu)
        mu_nb = tf.nn.softplus(mu)
        sigma_square = tf.add(tf.matmul(hidden_layer_out,
                                        self.decoder_weights_sigma_square),
                              self.decoder_bias_sigma_square)
        sigma_square = tf.clip_by_value(tf.nn.softplus(sigma_square),EPS, MAX_SIGMA_SQUARE)
        theta = tf.Variable(tf.random_normal([self.input_dim]))
        theta = tf.clip_by_value(tf.nn.softplus(theta),EPS, MAX_THETA)
        px_dropout = tf.add(tf.matmul(hidden_layer_out,
                                        self.decoder_weights_px_dropout),
                              self.decoder_bias_px_dropout)
        
        if args.recon_distribution == 'gaussian':
            param = LocationScale_decoder(mu,
                                          sigma_square,
                                          None,
                                          None)
        elif args.recon_distribution == 'student':
            param = LocationScale_decoder(mu,
                                         sigma_square,
                                         None,
                                         None)
        elif args.recon_distribution == 'nb':
            param = LocationScale_decoder(mu_nb,
                                          None,
                                          theta,
                                          None)
        elif args.recon_distribution == 'zinb':
            param = LocationScale_decoder(mu_nb,
                                          None,
                                          theta,
                                          px_dropout)
        return param

    
    # 完成整个编码器的过程，输出是 mu 和 sigma_square 两个参数
    def encoder(self, prob=0.9):
        weights_mu = tf.nn.dropout(self.weights_mu, prob) # 添加dropout噪声，设置神经元被选中的概率为0.9
        mu = tf.add(tf.matmul(self.hidden_layer_out, weights_mu),
                    self.bias_mu)
        sigma_square = tf.add(tf.matmul(self.hidden_layer_out,
                                        self.weights_sigma_square),
                              self.bias_sigma_square)

        return LocationScale_encoder(mu,
                                     tf.clip_by_value(tf.nn.softplus(sigma_square),
                                                      EPS, MAX_SIGMA_SQUARE))


# In[5]:

### model.py

import os
import numpy as np
from datetime import datetime

from matplotlib import pyplot as plt

# =============================================================================
class SCVIS(object):
    def __init__(self, architecture, hyperparameter):
        self.eps = 1e-20

        tf.reset_default_graph() # tf.reset_default_graph函数用于清除默认图形堆栈并重置全局默认图形
        # InteractiveSession是一种交互式的session方式，它让自己成为了默认的session，
        # 也就是说用户在不需要指明用哪个session运行的情况下，就可以运行起来，这就是默认的好处，
        # 这样的话就是run()和eval()函数可以不指明session。
        self.sess = tf.InteractiveSession()
        self.normalizer = tf.Variable(1.0, name='normalizer', trainable=False)

        self.architecture, self.hyperparameter = architecture, hyperparameter # 网络结构、超参数，都是字典类型
        self.regularizer_l2 = self.hyperparameter['regularizer_l2'] # 超参数中的L2正则化
        self.n = self.hyperparameter['batch_size'] # 超参数中的batch_size
        self.perplexity = self.hyperparameter['perplexity'] # 超参数中的困惑度 perplexity

        tf.set_random_seed(self.hyperparameter['seed']) # 超参数中的随机种子

        # Place_holders
        self.batch_size = tf.placeholder(dtype=tf.int32)
        self.x = tf.placeholder(tf.float32, shape=[None, self.architecture['input_dimension']]) # 输入数据
        self.z = tf.placeholder(tf.float32, shape=[None, self.architecture['latent_dimension']]) # 隐层变量

        self.p = tf.placeholder(tf.float32, shape=[None, None]) # 可能跟tsne的目标函数有关
        self.iter = tf.placeholder(dtype=tf.float32) # 迭代
        # vae为前面定义的高斯VAE
        self.vae = GaussianVAE(input_data=self.x,
                               input_size=self.batch_size,
                               layer_size=self.architecture['inference']['layer_size'],
                               output_dim=self.architecture['latent_dimension'],
                               decoder_layer_size=self.architecture['model']['layer_size'])
        
        # 把前面encoder返回的参数传进来
        self.encoder_parameter = self.vae.encoder_parameter
        self.latent = dict()
        self.latent['mu'] = self.encoder_parameter.mu
        self.latent['sigma_square'] = self.encoder_parameter.sigma_square
        self.latent['sigma'] = tf.sqrt(self.latent['sigma_square'])

        self.decoder_parameter = self.vae.decoder_parameter
        self.dof = tf.Variable(tf.constant(1.0, shape=[self.architecture['input_dimension']]),
                               trainable=True, name='dof')
        self.dof = tf.clip_by_value(self.dof, 0.1, 10, name='dof')
        
        # 关于ELBO的计算
        with tf.name_scope('ELBO'):
            self.weight = tf.clip_by_value(tf.reduce_sum(self.p, 0), 0.01, 2.0)
            
            if args.recon_distribution == 'gaussian':
                self.log_likelihood = tf.reduce_mean(
                        tf.multiply(log_likelihood_gaussian(self.x,
                                                self.decoder_parameter.mu,
                                                self.decoder_parameter.sigma_square),
                                    self.weight), name="log_likelihood")
            elif args.recon_distribution == 'student':
                self.log_likelihood = tf.reduce_mean(
                        tf.multiply(log_likelihood_student(self.x,
                                                self.decoder_parameter.mu,
                                                self.decoder_parameter.sigma_square,
                                                self.dof),
                                    self.weight), name="log_likelihood")
            elif args.recon_distribution == 'nb':
                self.log_likelihood = tf.reduce_mean(
                        tf.multiply(log_nb_positive(self.x,
                                            self.decoder_parameter.mu,
                                            self.decoder_parameter.theta),
                                    self.weight), name="log_likelihood")
            elif args.recon_distribution == 'zinb':
                self.log_likelihood = tf.reduce_mean(
                        tf.multiply(log_zinb_positive(self.x,
                                            self.decoder_parameter.mu,
                                            self.decoder_parameter.theta,
                                            self.decoder_parameter.px_dropout),
                                    self.weight), name="log_likelihood")

            self.kl_divergence = tf.reduce_mean(0.5 * tf.reduce_sum(self.latent['mu'] ** 2 + self.latent['sigma_square'] 
                                                                        - tf.log(self.latent['sigma_square']) - 1
                                                                    , reduction_indices=1)
                                                )
           # self.kl_divergence *= tf.maximum(0.1, self.architecture['input_dimension']/self.iter)
            self.elbo = self.log_likelihood - self.kl_divergence

        self.z_batch = self.vae.z

        # 关于非对称tsne目标函数KL散度的计算
        with tf.name_scope('tsne'):
            self.kl_pq = self.tsne_repel() * self.architecture['input_dimension'] # tf.minimum(self.iter, self.architecture['input_dimension'])
        
        # 最终的目标函数 这是最关键的地方！！！
        with tf.name_scope('objective'):
            self.obj = self.kl_pq + self.regularizer() - self.elbo
            #self.obj = self.regularizer() - self.elbo

        # Optimization 关于优化器的设置
        with tf.name_scope('optimizer'):
            learning_rate = self.hyperparameter['optimization']['learning_rate'] # 学习率
            
            # 可选择两种优化器
            if self.hyperparameter['optimization']['method'].lower() == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate)
            elif self.hyperparameter['optimization']['method'].lower() == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate,
                                                        beta1=0.9,
                                                        beta2=0.999,
                                                        epsilon=0.001)

            gradient_clipped = self.clip_gradient() #控制权重，防止梯度爆炸

            # 执行对应变量的更新梯度操作
            self.train_op = self.optimizer.apply_gradients(gradient_clipped, name='minimize_cost')

        self.saver = tf.train.Saver() # 保存模型
        
    # 控制权重，防止梯度爆炸
    def clip_gradient(self, clip_value=3.0, clip_norm=10.0):
        trainable_variable = self.sess.graph.get_collection('trainable_variables')
        grad_and_var = self.optimizer.compute_gradients(self.obj, trainable_variable)

        grad_and_var = [(grad, var) for grad, var in grad_and_var if grad is not None]
        grad, var = zip(*grad_and_var)
        grad, global_grad_norm = tf.clip_by_global_norm(grad, clip_norm=clip_norm)

        grad_clipped_and_var = [(tf.clip_by_value(grad[i], -clip_value*0.1, clip_value*0.1), var[i])
                                if 'encoder-sigma' in var[i].name
                                else (tf.clip_by_value(grad[i], -clip_value, clip_value), var[i])
                                for i in range(len(grad_and_var))]

        return grad_clipped_and_var
    
    # 定义L2正则化
    def regularizer(self):
        penalty = [tf.nn.l2_loss(var) for var in
                   self.sess.graph.get_collection('trainable_variables')
                   if 'weight' in var.name]

        l2_regularizer = self.regularizer_l2 * tf.add_n(penalty)

        return l2_regularizer
    
    # tsne目标函数中 KL散度的计算
    def tsne_repel(self):
        nu = tf.constant(self.architecture['latent_dimension'] - 1, dtype=tf.float32) ## 自由度 v = d-1

        sum_y = tf.reduce_sum(tf.square(self.z_batch), reduction_indices=1)  # 隐变量z的两个维度值相加，变成一个 1×128的矩阵，其中1是两个维度值的加和
        num = -2.0 * tf.matmul(self.z_batch,
                               self.z_batch,
                               transpose_b=True) + tf.reshape(sum_y, [-1, 1]) + sum_y
        # 这时 num 是一个 n×n 的 对称阵，对角线上元素为0，ij位置的元素值是第i个样本与第j个样本的距离，||z1-z2||^2  =  (z11 - z21)^2 + (z21 - z22)^2
        num = num / nu

        p = self.p + 0.1 / self.n
        p = p / tf.expand_dims(tf.reduce_sum(p, reduction_indices=1), 1)

        num = tf.pow(1.0 + num, -(nu + 1.0) / 2.0)
        attraction = tf.multiply(p, tf.log(num))
        attraction = -tf.reduce_sum(attraction) # 文中公式的第一项，也就是前面有负号的那一项

        den = tf.reduce_sum(num, reduction_indices=1) - 1
        repellant = tf.reduce_sum(tf.log(den)) # 文中公式的第二项

        return (repellant + attraction) / self.n  # 这里的 n=batch_size，也就是这一批次的样本数量
    
    # 这个函数后面用到，是想要返回每次迭代 elbo 和 tsne_cost 的值
    def _train_batch(self, x, t):
        p = compute_transition_probability(x, perplexity=self.perplexity) # 计算tsne距离的公式

        feed_dict = {self.x: x,
                     self.p: p,
                     self.batch_size: x.shape[0],
                     self.iter: t}

        _, elbo, tsne_cost , loss = self.sess.run([self.train_op, self.elbo, self.kl_pq, self.obj], feed_dict=feed_dict)

        return elbo, tsne_cost, loss
    # 训练
    def train(self, data, max_iter=1000, batch_size=None,
              pretrained_model=None, verbose=True, verbose_interval=10,
              show_plot=True, plot_dir='./img/'):

        max_iter = max_iter
        batch_size = batch_size or self.hyperparameter['batch_size']
        
        # 用于记录每次迭代的 elbo 和 tsne_cost的值
        status = dict()
        status['elbo'] = np.zeros(max_iter)
        status['tsne_cost'] = np.zeros(max_iter)
        status['loss'] = np.zeros(max_iter)

        if pretrained_model is None:
            self.sess.run(tf.global_variables_initializer()) # 初始化全局变量
        else:
            self.load_sess(pretrained_model) # 或者，读取之前训练过的模型
        
        start = datetime.now() # 记录训练开始时间
        for iter_i in range(max_iter):
            x, y = data.next_batch(batch_size)
            # 训练，并记录每次迭代 elbo 和 tsne_cost 的值
            status_batch = self._train_batch(x, iter_i+1)
            status['elbo'][iter_i] = status_batch[0]
            status['tsne_cost'][iter_i] = status_batch[1]
            status['loss'][iter_i] = status_batch[2]
            
            # 如果要展示训练过程 (verbose = True) 并且迭代次数为50的倍数，则打印当前迭代的信息，也就是每50次迭代打印一次信息
            if verbose and iter_i % verbose_interval == 0:
                print('Epoch = {} | Loss = {} | elbo = {} | scaled_tsne_cost = {}'.format(iter_i, status['loss'][iter_i], status['elbo'][iter_i], status['tsne_cost'][iter_i]))

        print('Time used for training: {}\n'.format(datetime.now() - start)) # 训练结束，打印训练时间
        return status

    # 编码，run起来
    def encode(self, x):
        var = self.vae.encoder(prob=1.0)
        feed_dict = {self.x: x}
        return self.sess.run(var, feed_dict=feed_dict)
    
    # 解码，run起来
    def decode(self, z):
        var = self.vae.decoder(tf.cast(z, tf.float32))
        feed_dict = {self.z: z, self.batch_size: z.shape[0]}

        return self.sess.run(var, feed_dict=feed_dict)
    
    def encode_decode(self, x):
        var = [self.latent['mu'],
               self.latent['sigma_square'],
               self.decoder_parameter.mu]
               #,self.decoder_parameter.sigma_square]

        feed_dict = {self.x: x, self.batch_size: x.shape[0]}

        return self.sess.run(var, feed_dict=feed_dict)
    
    def save_sess(self, model_name):
        self.saver.save(self.sess, model_name)

    def load_sess(self, model_name):
        self.saver.restore(self.sess, model_name)

    # 计算对数似然，用的是t分布的对数似然，前面定义的 log_likelihood
    def get_log_likelihood(self, x, dof=None):

        dof = dof or self.dof
        
        if args.recon_distribution == 'gaussian':
            log_likelihood = log_likelihood_gaussian(self.x,
                                                     self.decoder_parameter.mu,
                                                     self.decoder_parameter.sigma_square)
        elif args.recon_distribution == 'student':
            log_likelihood = log_likelihood_student(self.x,
                                                    self.decoder_parameter.mu,
                                                    self.decoder_parameter.sigma_square,
                                                    self.dof)
        elif args.recon_distribution == 'nb':
            log_likelihood = log_nb_positive(self.x,
                                             self.decoder_parameter.mu,
                                             self.decoder_parameter.sigma_square)
        elif args.recon_distribution == 'zinb':        
            log_likelihood = log_zinb_positive(self.x,
                                             self.decoder_parameter.mu,
                                             self.decoder_parameter.sigma_square,
                                             self.decoder_parameter.px_dropout)
        num_samples = 5

        feed_dict = {self.x: x, self.batch_size: x.shape[0]}
        log_likelihood_value = 0

        for i in range(num_samples):
            log_likelihood_value += self.sess.run(log_likelihood, feed_dict=feed_dict)

        log_likelihood_value /= np.float32(num_samples)

        return log_likelihood_value

    # 设置标准化
    def set_normalizer(self, normalizer=1.0):
        normalizer_op = self.normalizer.assign(normalizer) # tf.Variable.assign 重新给张量赋值
        self.sess.run(normalizer_op)

    # 实现标准化
    def get_normalizer(self):
        return self.sess.run(self.normalizer)

# In[7]:

### data.py

import numpy as np

MAX_VALUE = 1
LABEL = None

np.random.seed(0)


class DataSet(object):
    def __init__(self, x, y=LABEL, max_value=MAX_VALUE):
        if y is not None:
            # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发逗号后面定义的异常，这一句是判断x和y的行数是否相等
            assert x.shape[0] == y.shape[0], ('x.shape: %s, y.shape: %s' % (x.shape, y.shape))

        self._num_data = x.shape[0] # 样本数
        x = x.astype(np.float32) # 将 x 转化成 float32
        x /= max_value

        self._x = x
        self._y = y
        self._epoch = 0
        self._index_in_epoch = 0

        index = np.arange(self._num_data) # [0, 1, ... _num_data]
        np.random.shuffle(index) # 打乱样本索引
        self._index = index # 新的样本索引 比如 如果原来索引是 [0, 1, 2, 3, 4] 现在可能是[3, 1, 0, 4, 2]

    # @property 是修饰方法，会将方法转换为相同名称的只读属性,可以与所定义的属性配合使用，这样可以防止属性被修改。
    @property
    def all_data(self): # 所有数据
        return self._x

    @property
    def label(self):    # 所有数据的标签
        return self._y

    @property
    def num_of_data(self): # 所有数据的数量
        return self._num_data

    @property
    def completed_epoch(self):  # 训练的轮次
        return self._epoch

    # 记录一个batch中数据的起始索引和终止索引值，返回batch_size个数据及其标签
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        # 如果索引超出了数据总数 则进入下一个epoch，并重新令start为0
        if self._index_in_epoch > self._num_data: 
            # 如果 batch_size 大于 数据总数，则抛出错误
            assert batch_size <= self._num_data, ('batch_size: %s, num_data: %s' % (batch_size, self._num_data))

            self._epoch += 1
            np.random.shuffle(self._index)

            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch

        index = self._index[start:end]
        if self._y is not None:
            y = self._y[index]
        else:
            y = self._y

        return self._x[index], y

# In[8]:

### run.py

#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import yaml # yaml是一个专门用来写配置文件的语言
import pandas as pd


CURR_PATH = os.path.dirname(os.path.realpath('__file__')) # 获取当前路径


def train(args):

    x, y, architecture, hyperparameter, train_data, model, normalizer, out_dir, name = _init_model(args, 'train')
    print('-------数据：{} × {}---------------------'.format(np.shape(x)[0], np.shape(x)[1]))

    max_iter = hyperparameter['max_epoch'] # 自己改的
    name += '_iter_' + str(max_iter) # 后面用作输出图的命名
    # 训练，并设定相应的参数
    res = model.train(data=train_data,
                      batch_size=hyperparameter['batch_size'],
                      verbose=args.verbose,
                      verbose_interval=args.verbose_interval,
                      show_plot=args.show_plot,
                      plot_dir=os.path.join(out_dir, (name+"_intermediate_result")),
                      max_iter=max_iter,
                      pretrained_model=args.pretrained_model_file)
    model.set_normalizer(normalizer) # 设置标准化，前面的函数看的不太懂

    # Save the trained model
    out_dir = args.out_dir
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    model_dir = os.path.join(out_dir, "model")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    model_name = name + ".ckpt"
    model_name = os.path.join(model_dir, model_name)
    model.save_sess(model_name)

    # The objective function trace plot
    loss = res['loss']
    iteration = len(loss)
    plt.plot(list(range(iteration)), loss, label='loss' )
    plt.title("Loss", fontsize=16)
    plt.xlabel("number of epochs", fontsize=12)
    plt.ylabel("objective function",fontsize=12)
    plt.legend(fontsize=14)
    plt.tight_layout()

    # Save the mapping results
    _save_result(x, y, model, out_dir, name)

    return()


# 初始模型，为模型传入数据、结构和输出保存路径
def _init_model(args, mode):
    ##x = pd.read_csv(args.data_matrix_file, sep='\t').values # 读取数据
    x = pd.read_csv(args.data_matrix_file, sep=',', index_col=0).values # 读取数据
# =============================================================================
#     x = np.array(x, dtype=np.int)
#     selected = np.std(x, axis=0).argsort()[-1000:][::-1]
#     x = x[:, selected]
# =============================================================================
    x = np.log(x + 1)
    config = {}
    config_file = CURR_PATH + '/config/model_config.yaml' # 读取配置好的参数
    config_file = args.config_file or config_file
    try:
        config_file_yaml = open(config_file, 'r')
        config = yaml.load(config_file_yaml)
        config_file_yaml.close()
    except yaml.YAMLError as exc:
        print('Error in the configuration file: {}'.format(exc))

    architecture = config['architecture'] # 模型的结构
    architecture.update({'input_dimension': x.shape[1]})

    hyperparameter = config['hyperparameter']
    if hyperparameter['batch_size'] > x.shape[0]:
        hyperparameter.update({'batch_size': x.shape[0]})

    model = SCVIS(architecture, hyperparameter) # model定义为SCVIS大类
    normalizer = 1.0
    if args.pretrained_model_file is not None:  # 预训练数据
        model.load_sess(args.pretrained_model_file)
        normalizer = model.get_normalizer()

    if mode == 'train':
        if args.normalize is not None:
            normalizer = float(args.normalize)
        else:
            normalizer = np.max(np.abs(x))
    else:
        if args.normalize is not None:
            normalizer = float(args.normalize)

    # x /= normalizer

    y = None
    if args.data_label_file is not None:
        ##label = pd.read_csv(args.data_label_file, sep='\t').values
        ##label = pd.Categorical(label[:, 0])
        label = pd.read_csv(args.data_label_file, sep=',', index_col=0)['cell_type'].values
        label = pd.Categorical(label)
        y = label.codes

    # fixed random seed
    np.random.seed(0)
    train_data = DataSet(x, y)

    out_dir = args.out_dir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # 输出文件的名字
    name = '_'.join(['perplexity', str(hyperparameter['perplexity']),
                     'regularizer', str(hyperparameter['regularizer_l2']),
                     'batch_size', str(hyperparameter['batch_size']),
                     'learning_rate', str(hyperparameter['optimization']['learning_rate']),
                     'latent_dimension', str(architecture['latent_dimension']),
                     'activation', str(architecture['activation']),
                     'seed', str(hyperparameter['seed'])])

    return x, y, architecture, hyperparameter, train_data, model, normalizer, out_dir, name


from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI

def show_tSNE(latent, labels, return_tSNE=False):
    
    if latent.shape[1] != 2:
        latent = TSNE().fit_transform(latent)
    
    latent_df = pd.DataFrame(latent, columns=['tsne1','tsne2'])
    latent_df['labels'] = labels
    unique_label = np.unique(labels)
    colors=['grey','gold','darkviolet','turquoise','r','g','salmon', 'c','b','m','y','darkorange','lightgreen','plum', 'tan','khaki', 'pink', 'skyblue','lawngreen','k']
    
    labels_pred = KMeans(n_clusters=len(unique_label)).fit_predict(latent)
    print("silhouette_score =", silhouette_score(latent, labels))
    print("ARI =", ARI(labels, labels_pred))
    print("NMI =", NMI(labels, labels_pred))
    
    plt.figure(figsize=(10, 8))
    for i, v in enumerate(unique_label):
        plt.scatter(latent_df['tsne1'][latent_df['labels']==v], latent_df['tsne2'][latent_df['labels']==v],
                    label=v, c=colors[i], edgecolors='none', s=20)
    plt.legend(prop = {'size':14}, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

    #plt.figure(figsize=(8, 8)) 
    #plt.scatter(latent[:, 0], latent[:, 1], c=labels, cmap=cmap, edgecolors='none')
    plt.axis("off")
    plt.tight_layout()

    if return_tSNE:
        return latent


# 将输出结果保存为文件
def _save_result(x, y, model, out_dir, name):
    x_hat = model.encode_decode(x)[2]
    x_hat = pd.DataFrame(x_hat)
    
    x_hat.to_csv("x_hat_mouse_filteredcell2_" + args.recon_distribution + ".csv")
    
    z_mu, _ = model.encode(x)
    print('-------------shape(z) = {} × {} ----------------'.format(np.shape(z_mu)[0], np.shape(z_mu)[1]))
    ct = pd.read_csv(args.data_label_file, sep=',', index_col=0)['cell_type'].values
    tsne = show_tSNE(z_mu, ct, return_tSNE=True)
    
# =============================================================================
#     ## 批次效应散点图
#     batch = pd.read_csv('../mouse_cell_atlas_filteredcell_labels.csv',index_col=0)['batch'].values
#     plt.figure(figsize=(8, 8))
#     plt.scatter(tsne[:,0],tsne[:,1], c=batch, cmap=plt.get_cmap('brg', 2), edgecolors='none', s=20)
#     plt.axis("off")
#     plt.tight_layout()
# =============================================================================


# In[9]:


class analysis_parser():
    def __init__(self):
        self.data_matrix_file = '../datases/mouse_cell_atlas_filteredcell2_2000.csv'
        self.config_file = 'model_config.yaml'  ## 这里面改隐空间的维度参数
        self.pretrained_model_file = None
        self.normalize = None
        self.data_label_file = '../datasets/mouse_cell_atlas_filteredcell2_labels.csv'
        self.verbose = 1
        self.verbose_interval = 10
        self.show_plot = 0
        self.out_dir = './'
        self.recon_distribution = 'zinb' ## 这里修改为gaussian,student,nb,zinb


# In[11]:
        
args = analysis_parser()

mode = 'train'
train(args)

