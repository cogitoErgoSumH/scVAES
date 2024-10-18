"""
Code for the paper single-cell Variational Inference (scVI) paper

Romain Lopez, Jeffrey Regier, Michael Cole, Michael Jordan, Nir Yosef
EECS, UC Berkeley

"""


import functools
import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim


# 定义层数结构的函数
def dense(x, 
          num_outputs,
          STD=0.01,
          keep_prob=None,
          activation=None,
          bn=False,
          phase=None):
    """
    Defining the elementary layer of the network

    Note:
    We adjust the standard deviation of the weight initialization to 0.01. This is useful considering the large counts in the data and guarantees numerical stability of the algorithm.  
    ## 我们调整权重初始化的标准差为0.01。这在考虑数据量大的情况下是有用的，并保证了算法的数值稳定性。
    Batchnorm paper: https://arxiv.org/abs/1502.03167 ## 批次标准化的原始文献
    Dropout paper: http://jmlr.org/papers/v15/srivastava14a.html ## dropout的原始文献

    Variables:
    x: tensorflow variable ## 数据
    num_outputs: number of outputs neurons after the dense layers  ## 输出神经元的个数
    keep_prob: float number for probability of keeping an individual neuron for dropout layer  ## 浮点数，丢失层保留单个神经元的概率
    activation: tensorflow activation function (tf.exp, tf.nn.relu...) for this layer ## 这一层的激活函数，可以是 tf.exp, tf.nn.relu 等
    bn: bool to use batchnorm for this layer  # 布尔值，是否使用批次标准化，默认False
    phase: tensorflow boolean node indicating whether training of testing phase (see dropout and batchnorm paper) ## 表示测试阶段是否进行训练？？
    """
    output = tf.identity(x) # 就是返回一个和x一样的新的tensor
    
    if keep_prob is not None: # 如果传入了参数 keep_prob ，则按照概率keep_prob进行dropout处理
        output = tf.layers.dropout(output, rate=keep_prob, training=phase)
        
    output = slim.fully_connected(output, num_outputs, activation_fn=None, \
                            weights_initializer=tf.truncated_normal_initializer(stddev=STD))
    # 添加一个全连接层，输入数据为 output， num_outputs 是层中输出单元的数量，返回的是表示一系列运算结果的张量变量 shape(output) = [batch_size, num_outputs]

    if bn: # 如果 bn=True 则进行批次标准化处理
        output = tf.layers.batch_normalization(output, training=phase)
        
    if activation:  # 如果传入了activation 则进行激活函数处理
        output = activation(output)
            
    return output


# 产生对角协方差的多元高斯分布的样本
def gaussian_sample(mean, var, scope=None):
    """
    Function to sample from a multivariate gaussian with diagonal covariance in tensorflow

    Note:
    This layer can either be parametrized by the variance or the log variance in a variational autoencoder. 
    We found by trials that it does not matter much
    ## 注意：这一层可以通过变分自编码器中的方差或对数方差来进行参数化，我们通过实验发现这并不重要
    
    Variables: #变量：
    mean: tf variable indicating the minibatch mean (shape minibatch_size x latent_space_dim)  # 均值：minibatch的均值 shape = (minibatch, latent_space_dim)
    var: tf variable indicating the minibatch variance (same shape) #  方差: minibatch的方差
    """
    with tf.compat.v1.variable_scope(scope, 'gaussian_sample'):
        sample = tf.random.normal(tf.shape(mean), mean, tf.sqrt(var))
        sample.set_shape(mean.get_shape())
        return sample

    
## 根据ZINB模型，minibatch的对数似然（标量）
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

    print("in log_zinb_positive, type(x) = ", type(x))
    print("in log_zinb_positive, shape(x) = ", x.shape)
    print("in log_zinb_positive, type(n) = ", type(n))
    print("in log_zinb_positive, type(n) = ", type(d))
    print("in log_zinb_positive, n = ", n)
    print("in log_zinb_positive, d = ", d)
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
    return tf.reduce_sum(res, axis=-1)

## 根据NB模型，minibatch的对数似然（标量）
def log_nb_positive(x, mu, theta, eps=1e-8):
    """
    log likelihood (scalar) of a minibatch according to a nb model. 
    
    Variables:
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    eps: numerical stability constant
    """    
    res = tf.lgamma(x + theta) - tf.lgamma(theta) - tf.lgamma(x + 1) + x * tf.log(mu + eps) \
                                - x * tf.log(theta + mu + eps) + theta * tf.log(theta + eps) \
                                - theta * tf.log(theta + mu + eps)
    return tf.reduce_sum(res, axis=-1)

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    ## 修饰器，如果没有提供参数，允许使用不带圆括号的装饰器，所有的参数必须是可选的
    
    Notes:
    https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    """
    # Python装饰器（decorator）在实现的时候，被装饰后的函数其实已经是另外一个函数了（函数名等函数属性会发生改变），
    # 为了不影响，Python的functools包中提供了一个叫wraps的decorator来消除这样的副作用。
    # 写一个decorator的时候，最好在实现之前加上functools的wrap，它能保留原有函数的名称和函数属性
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]): # callable函数用于检查一个对象是否是可调用的
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    ## 一个用于定义TensorFlow操作的函数的装饰器。被包装的函数将只执行一次。
    ## 后续对它的调用将直接返回结果，这样操作只会添加到图中一次。
    ## 函数添加的操作都在tf.variable_scope()中。
    ## 如果这个装饰器与参数一起使用，它们将被转发到变量作用域。作用域名称默认为被包装函数的名称。
    
    Notes:
    https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    # @property装饰器会将方法转换为相同名称的只读属性，可以与所定义的属性配合使用，这样可以防止属性被修改。
    # 加了@property后，可以用调用属性的形式来调用方法,后面不需要加()
    # 通过@property的方法，可以隐藏属性名，让用户进行使用的时候无法随意修改。
    @property  
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):    # hasattr() 函数用于判断对象是否包含对应的属性。
            with tf.compat.v1.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))    # setattr() 函数=用于设置属性值，该属性不一定是存在的。
        return getattr(self, attribute) #  getattr() 函数用于返回一个对象属性值
    return decorator

# 以下两个函数为MMD方法相关，二选一
def mmd_fourier(x1, x2, bandwidth=2., dim_r=500):
    """
    Approximate RBF kernel by random features
    ## 用随机特征近似RBF核

    Notes:
    Reimplementation in tensorflow of the Variational Fair Autoencoder
    https://arxiv.org/abs/1511.00830
    """
    d = x1.get_shape().as_list()[1]
    rW_n = tf.sqrt(2. / bandwidth) * tf.random_normal([d, dim_r]) / np.sqrt(d)
    rb_u = 2 * np.pi * tf.random_uniform([dim_r])
    rf0 = tf.sqrt(2. / dim_r) * tf.cos(tf.matmul(x1, rW_n) + rb_u)
    rf1 = tf.sqrt(2. / dim_r) * tf.cos(tf.matmul(x2, rW_n) + rb_u)
    result = tf.reduce_sum((tf.reduce_mean(rf0, axis=0) - tf.reduce_mean(rf1, axis=0))**2)
    return tf.sqrt(result)

def mmd_rbf(x1, x2, bandwidths=1. / (2 * (np.array([1., 2., 5., 8., 10])**2))):
    """
    Return the mmd score between a pair of observations
    # 返回一对观察结果之间的mmd得分
    
    Notes:
    Reimplementation in tensorflow of the Variational Fair Autoencoder
    https://arxiv.org/abs/1511.00830
    """
    d1 = x1.get_shape().as_list()[1]
    d2 = x2.get_shape().as_list()[1]
    
    def K(x1, x2, gamma=1.): 
        dist_table = tf.expand_dims(x1, 0) - tf.expand_dims(x2, 1)
        return tf.transpose(tf.exp(-gamma * tf.reduce_sum(dist_table **2, axis=2)))

    # possibly mixture of kernels
    x1x1, x1x2, x2x2 = 0, 0, 0
    for bandwidth in bandwidths:
        x1x1 += K(x1, x1, gamma=np.sqrt(d1) * bandwidth) / len(bandwidths)
        x2x2 += K(x2, x2, gamma=np.sqrt(d2) * bandwidth) / len(bandwidths)
        x1x2 += K(x1, x2, gamma=np.sqrt(d1) * bandwidth) / len(bandwidths)

    return tf.sqrt(tf.reduce_mean(x1x1) - 2 * tf.reduce_mean(x1x2) + tf.reduce_mean(x2x2))

# 定义mmd正则项的函数
def mmd_objective(z, s, sdim):
    """
    Compute the MMD from latent space and nuisance_id
    # 从潜在空间和nuisance_id计算MMD
    
    Notes:
    Reimplementation in tensorflow of the Variational Fair Autoencoder
    https://arxiv.org/abs/1511.00830
    """
    
    #mmd_method = mmd_rbf
    mmd_method = mmd_fourier
    
    z_dim = z.get_shape().as_list()[1]

    # STEP 1: construct lists of samples in their proper batches
    z_part = tf.dynamic_partition(z, s, sdim)

                
    # STEP 2: add noise to all of them and get the mmd
    mmd = 0
    for j, z_j in enumerate(z_part):
        z0_ = z_j
        aux_z0 = tf.random_normal([1, z_dim])  # if an S category does not have any samples
        z0 = tf.concat([z0_, aux_z0], 0)
        if len(z_part) == 2:
            z1_ = z_part[j + 1]
            aux_z1 = tf.random_normal((1, z_dim))
            z1 = tf.concat([z1_, aux_z1], axis=0)
            return mmd_method(z0, z1)
        z1 = z
        mmd += mmd_method(z0, z1)
    return mmd


class scVIModel:

    def __init__(self, expression=None, batch_ind=None, num_batches=None, kl_scale=None, mmd_scale=None, phase=None,\
                 library_size_mean = None, library_size_var = None, apply_mmd=False, \
                 dispersion="gene", n_layers=1, n_hidden=128, n_latent=10, \
                 dropout_rate=0.1, log_variational=True, optimize_algo=None, zi=True):
        """
        Main parametrization of the scVI algorithm.

        Notes and disclaimer: # 声明
        + We recommend to put kl_scale to 1 for every tasks except clustering where 0 will lead better discrepency between the clusters
        ## 除了聚类，建议将kl_scale设置为1，因为0会使得簇之间的差异更大
        + Applying a too harsh penalty will ruin your biology info. We recommend using less than a 100. From ongoing tests, using zero actually removes batch effects as well as the paper results.
        ## 过于严厉的惩罚会破坏你的生物信息，建议mmd_scale<100，从正在进行的测试来看，使用0不能实现批处理效果和论文中的结果。
        + We recommend the dispersion parameter to be gene specific (or batch-specific as in the paper) as in ZINB-WaVE if you do not have enough cells
        ## 如果没有足够的细胞，我们建议dispersion参数是基因特异性的(或如论文中所述的批次特异性的)
        + To better remove library size effects between clusters, mention the log-library size prior for each batch (like in the paper)
        ## 为了更好地消除簇之间的文库大小影响，在每个批处理之前使用对数库大小(就像在论文中那样)

        Variables:
        expression: tensorflow variable of shape (minibatch_size x genes), placeholder for input counts data  ## shape=minibatch×genes
        batch_ind: tensorflow variable for batch indices (minibatch_size) with integers from 0 to n_batches - 1  ## 第几批，整数，范围从0到n_batches-1
        kl_scale: tensorflow variable for scalar multiplier of the z kl divergence ## 变量Z的KL散度的标量乘子
        mmd_scale: tensorflow variable for scalar multiplier of the MMD penalty ## maximum mean discrepancy，最大平均差异，MMD惩罚项的标量乘子
        phase: tensorflow variable for training phase ## 表示训练阶段
        library_size_mean = either a number or a list for each batch of the mean log library size ## 数字或列表，表示每一批对数文库大小的均值，毕业论文中是 ξ
        library_size_var = either a number or a list for each batch of the variance of the log library size ## 数字或列表，表示每一批对数文库大小的方差，毕业论文中是 τ
        apply_mmd: boolean to choose whether to use a MMD penalty ## 布尔值变量，表示是否使用MMD惩罚项
        dispersion: "gene" (n_genes params) or "gene-batch" (n_genes x n_batches params) or "gene-cell" (a neural nets) ## 这个参数上面有说明，使用"gene"
        n_layers: a integer for the number of layers in each neural net. We use 1 throughout the paper except on the 1M dataset where we tried (1, 2, 3) hidden layers
                    ## 整数，表示神经网络中的（隐藏层？）层数，在整片文章中都使用1，除了在1M数据集中使用(1,2,3)隐藏层
        n_hidden: number of neurons for each hidden layer. Always 128. ## 每个隐藏层的神经元个数，固定为128
        n_latent: number of desired dimension for the latent space ## 隐空间维数，指的是隐变量Z的维度
        dropout_rate: rate to use for the dropout layer (see elementary layer function). always 0.1 ## dropout rate ，固定为0.1
        log_variational: whether to apply a logarithmic layer at the input of the variational network (for < 4000 cells datasets) 
                        ## 是否在变分网络的输入端应用对数层(对于小于4000细胞数据集)，默认True
        optimize_algo: a tensorflow optimizer ## 优化器
        zi: whether to use a ZINB or a NB distribution ## 使用ZINB或NB分布
        """
        
        # Gene expression placeholder
        if expression is None: # 如果没有传入expression，则提示需提供一个表达数据
            raise ValueError("provide a tensor for expression data")
        self.expression = expression
        
        print("Running scVI on "+ str(self.expression.get_shape().as_list()[1]) + " genes") # 打印基因数（列数）
        self.log_variational = log_variational # 默认True

        # batch correction ## 批次校正
        if batch_ind is None: # 如果 batch_ind 没有传入参数，
            print("scVI will run without batch correction") # 模型可以运行，但不进行批次校正
            self.batch = None       # 变量 batch 为 None
            self.apply_mmd = False # 不使用MMD惩罚项
            
        else:                 # 如果 batch_ind 传入了参数
            if num_batches is None: # 而 num_batches 没有传入参数，则抛出以下报错
                raise ValueError("provide a comprehensive list of unique batch ids") # 需提供一个完整唯一的批次id列表
            # 如果 num_batches 传入了参数，则为以下变量赋值
            self.batch_ind = batch_ind
            self.num_batches = num_batches
            self.batch = tf.one_hot(batch_ind, num_batches)  
            # tf.one_hot(indices, depth) ; indices：输入的张量[batch, features]; depth：onehot编码的一个维度; 输出矩阵默认为 batch × features × depth
            # tf.one_hot()函数规定输入的元素indices从0开始，最大的元素值不能超过（depth - 1），因此能够表示depth个单位的输入。若输入的元素值超出范围，输出的编码均为 [0, 0 … 0, 0]
            # 例：tf.one_hot(indices=[2,4,0,3], depth=5) = [[0,0,1,0,0],[0,0,0,0,1],[1,0,0,0,0],[0,0,0,1,0]]
            # 在 Zeisel_dataset 的例子中，batch_ind = b_train = [1,0,1,...,1,0,1] ， num_batches = 2，
            # 所以 self.batch = [[0,1],[1,0],[0,1],...,[0,1],[1,0],[0,1]] ，shape = (2253,2)
            self.mmd_scale = mmd_scale
            self.apply_mmd = apply_mmd

            print("Got " + str(num_batches) + " batches in the data")
            if self.apply_mmd: # 是否使用MMD惩罚项
                print("Will apply a MMD penalty")
            else: 
                print("Will not apply a MMD penalty")
        
        #kl divergence scalar ## 需要提供KL散度的标量乘子，否则报错
        if kl_scale is None:
            raise ValueError("provide a tensor for kl scalar")
        self.kl_scale = kl_scale
                
        #prior placeholder ## 需要传入文库大小的均值和方差，否则报错
        if library_size_mean is None or library_size_var is None:
            raise ValueError("provide prior for library size")
            
        if type(library_size_mean) in [float, np.float64] : # 如果文库大小的均值是浮点型
            self.library_mode = "numeric"
            #self.library_size_mean = tf.to_float(tf.constant(library_size_mean))
            self.library_size_mean = tf.cast(tf.constant(library_size_mean), float)
            self.library_size_var = tf.cast(tf.constant(library_size_var), float)
            
        else: # 如果文库大小的均值不是浮点型
            if library_size_mean.get_shape().as_list()[0] != num_batches: # 如果文库大小的均值的长度不等于批次数量，则报错
                raise ValueError("provide correct prior for library size (check batch shape)") # "为库大小提供正确的先验(检查batch shape)"
            else: # 如果文库大小的均值的长度等于批次数量
                self.library_mode = "list"
                self.library_size_mean = library_size_mean
                self.library_size_var = library_size_var 
                
        print("Will work on mode " + self.library_mode + " for incorporating library size") # 在library_mode模式下合并library size
        
        # high level model parameters
        if dispersion not in ["gene", "gene-batch", "gene-cell"]:
            raise ValueError("dispersion should be in gene / gene-batch / gene-cell")
        self.dispersion = dispersion
        
        print("Will work on mode " + self.dispersion + " for modeling inverse dispersion param") # 在dispersion模式下对逆dispersion参数建模
        
        self.zi = zi  # 默认为True，使用ZINB分布
        if zi:
            print("Will apply zero inflation")
        
        # neural nets architecture ## 神经网络架构！！！
        self.n_hidden = n_hidden  # 每个隐藏层的神经元个数，默认为128
        self.n_latent = n_latent  # 隐空间维数，指的是隐变量Z的维度，默认为10
        self.n_layers = n_layers  # 隐藏层数，默认为1
        self.n_input = self.expression.get_shape().as_list()[1] ### 输入节点数，即列数 or 特征数 or 基因数

        print(str(self.n_layers) + " hidden layers at " + str(self.n_hidden) + " each for a final " + str(self.n_latent) + " latent space")
        # 1 hidden layers at 128 each for a final 10 latent space，1个隐藏层，每层128个神经元，最后的隐空间为10维
        
        # on training variables
        self.dropout_rate = dropout_rate  # 默认为0.1
        # 训练阶段 phase 不能为 None
        if phase is None:  
            raise ValueError("provide an optimization metadata (phase)")
        self.training_phase = phase
        # 优化器不能为None
        if optimize_algo is None:
            raise ValueError("provide an optimization method")
        self.optimize_algo = optimize_algo
        
        # call functions ## 调用函数，因为使用了@修饰符，可以不加括号，直接以属性的方式调用
        self.variational_distribution
        self.sampling_latent
        self.generative_model
        self.optimize
        self.optimize_test
        self.imputation
    
    # 编译器读到 @define_scope，去调用 define_scope 函数，define_scope 函数的入口参数是 variational_distribution() 这个函数
    # 1. 函数先定义，再修饰它；反之会编译器不认识；
    # 2. 修饰符“@”后面必须是之前定义的某一个函数；
    # 3. 每个函数只能有一个修饰符，大于等于两个则不可以。
    @define_scope
    def variational_distribution(self): # 相当于 define_scope(variational_distribution())
        """
        defines the variational distribution or inference network of the model
        ## 定义模型的变分分布或推断网络
        q(z, l | x, s)


        """

        #q(z | x, s) ！！！
        if self.log_variational: # 默认True
            x = tf.math.log(1 + self.expression) # log(1+x)
        else:
            x = self.expression

        h = dense(x, self.n_hidden, activation=tf.nn.relu, \
                    bn=True, keep_prob=self.dropout_rate, phase=self.training_phase)
        # x → n_hidden ，即 genes → 128，编码过程
        for layer in range(2, self.n_layers + 1):
            h = dense(h, self.n_hidden, activation=tf.nn.relu, \
                bn=True, keep_prob=self.dropout_rate, phase=self.training_phase)
        # 默认 n_layers = 1 ，所以不会进行这个循环，如果 n_layers > 1 则有若干次 128 → 128
        
        self.qz_m = dense(h, self.n_latent, activation=None, \
                bn=False, keep_prob=None, phase=self.training_phase)
        self.qz_v = dense(h, self.n_latent, activation=tf.exp, \
                bn=False, keep_prob=None, phase=self.training_phase)
        ## 得到q(z | x, s)的两个参数，均值和方差，维数为 n_latent ，即10维。均值不用激活函数，方差用 tf.exp 激活函数。
        
        # q(l | x, s) ！！！
        h = dense(x, self.n_hidden, activation=tf.nn.relu, \
                bn=True, keep_prob=self.dropout_rate, phase=self.training_phase)
        # 相比上面，少了循环那一步
        self.ql_m = dense(h, 1, activation=None, \
                bn=False, keep_prob=None, phase=self.training_phase)
        self.ql_v = dense(h, 1, activation=tf.exp, \
                bn=False, keep_prob=None, phase=self.training_phase)
        ## 同理，得到 q(l | x, s)的两个参数，均值和方差，都是一维的。均值不用激活函数，方差用 tf.exp 激活函数确保为正。
    
    @define_scope
    def sampling_latent(self):
        """
        defines the sampling process on the latent space given the var distribution
        ## 定义在给定方差的分布在隐空间中的采样过程
        """
            
        self.z = gaussian_sample(self.qz_m, self.qz_v)
        self.library = gaussian_sample(self.ql_m, self.ql_v)  ## self.library = log l
        ## z和l都是从高斯分布中采样，这两个高斯分布的均值和方差由variational_distribution给出

    @define_scope
    def generative_model(self):
        """
        defines the generative process given a latent variable (the conditional distribution)
        ## 定义了给定隐变量的生成过程（条件分布）
        """
            
        # p(x | z, s) ！！！
        if self.batch is not None:  # 传入了参数 batch_ind
            h = tf.concat([self.z, self.batch], 1) # 此时 self.batch 为 one_hot 变量
            # shape(z) = [batch_size, 10] ， shape(batch)  = [batch_size, num_batches]， shape(h) = [batch_size, 10 + num_batches] = [batch_size, 12]
        else:
            h = self.z # shape(h) = [batch_size, 10]
        
        #h = dense(h, self.n_hidden,
        #          activation=tf.nn.relu, bn=True, keep_prob=self.dropout_rate, phase=self.training_phase)
        h = dense(h, self.n_hidden,
                  activation=tf.nn.relu, bn=True, keep_prob=None, phase=self.training_phase)
        # h[1] → 128，即 10 → 128，解码过程
        
        for layer in range(2, self.n_layers + 1):
            if self.batch is not None:
                h = tf.concat([h, self.batch], 1)
            h = dense(h, self.n_hidden, activation=tf.nn.relu, \
                bn=True, keep_prob=self.dropout_rate, phase=self.training_phase)
        # 默认不进行循环，逐层解码过程
        
        if self.batch is not None:
            h = tf.concat([h, self.batch], 1)   ## 128 或 128 + num_batches 维     128 + 2 = 130 维
        
        # mean gamma ## 原文中的ρ·θ，毕业论文里面修改了这里为αβ
        self.px_scale = dense(h, self.n_input, activation=tf.nn.softmax, \
                    bn=False, keep_prob=None, phase=self.training_phase)
        # 128 → n_input 解码回到输入维数，一个系数，激活函数用 softmax，这是论文中的神经网络fw，得到Gamma分布的参数ρ·θ
        
        # dispersion ## NB分布的参数 logθ’，不是 原文 w~Gamma(ρ,θ)中的θ，毕业论文中改了
        if self.dispersion == "gene-cell":
            self.px_r = dense(h, self.n_input, activation=None, \
                    bn=False, keep_prob=None, phase=self.training_phase)
        elif self.dispersion == "gene": ## 默认情况
            self.px_r = tf.Variable(tf.random.normal([self.n_input]), name="r") # px_r = logθ
            # 维度为 n_input， tf.random.normal(shape, mean=0, stddev=1)默认从标准正态分布中随机取出张量形状如 shape 的值
        else:
            if self.batch_ind is None:
                raise ValueError("batch dispersion with no batch info")
            else:
                self.px_r = tf.Variable(tf.random_normal([self.num_batches, self.n_input]), name="r")
        # 根据 dispersion ，三种 px_r 的计算方法，默认用第二种
            
        # mean poisson  y ~ Poisson(lw)
        self.px_rate = tf.exp(self.library) * self.px_scale  ## exp(library) = exp(log l ) = l ， px_rate 对应论文Note3中的 l·fw(z,s)

        #dropout
        if self.zi: # 使用 ZINB 分布
            self.px_dropout = dense(h, self.n_input, activation=None, \
                    bn=False, keep_prob=None, phase=self.training_phase) 
        # 128 → n_input，这是论文中的神经网络 fh，表示 dropout的概率的 logit，论文中的参数pi
        # 这里没有加一个限制在[0,1]之间的激活函数，是因为 px_dropout 是 logit(pi)

    @define_scope
    def optimize(self):
        """
        write down the loss and the optimizer
        """
        
        # converting from batch to local quantities ## 从批量转换为本地数量，看不懂
        if self.dispersion == "gene-batch":
            local_dispersion = tf.matmul(self.batch, tf.exp(self.px_r))
        else: ## 默认情况
            local_dispersion = tf.exp(self.px_r)  # exp(logθ) = θ
            
        if self.library_mode == "numeric":
            local_l_mean = self.library_size_mean # l的均值，是传入的常参数，毕业论文中是 ξ
            local_l_var = self.library_size_var # l的方差，是传入的常参数，毕业论文中是 τ^2
        else:
            local_l_mean = tf.matmul(self.batch, self.library_size_mean)
            local_l_var = tf.matmul(self.batch, self.library_size_var)
        
        
        # VAE loss 重构误差！！
        if self.zi: ## 默认情况
            recon = log_zinb_positive(self.expression, self.px_rate, local_dispersion, \
                                  self.px_dropout)
            # recon = log_zinb_positive(x, mu, theta, pi) 即 log p( x | z,l,s )
            # mu = l·fw = l·ρ； theta = exp(logθ) = θ； pi = fh
        else:
            recon = log_nb_positive(self.expression, self.px_rate, local_dispersion)
                    # log_nb_positive(x, mu, theta)
        kl_gauss_z = 0.5 * tf.reduce_sum(\
                        tf.square(self.qz_m) + self.qz_v - tf.log(1e-8 + self.qz_v) - 1, 1)   # KL[ q(z|x,s) || p(z) ]
        kl_gauss_l = 0.5 * tf.reduce_sum(\
                        tf.square(self.ql_m - local_l_mean) / local_l_var  \
                            + self.ql_v / local_l_var \
                            + tf.log(1e-8 + local_l_var)  - tf.log(1e-8 + self.ql_v) - 1, 1)  # KL[ q(l|x,s) || p(l) ]
        
        # ELBO，变分下界！！
        self.ELBO_gau = tf.reduce_mean(recon - self.kl_scale * kl_gauss_z - kl_gauss_l) # KL[ q(z|x,s) || p(z) ] 前有一个系数
    
        # MMD loss，默认不使用
        if self.apply_mmd:  # 如果使用MMD 则加上前面定义好的 mmd_objective 函数，作为惩罚项
            self.mmd = mmd_objective(self.z, self.batch_ind, self.num_batches)
            self.loss = - self.ELBO_gau + self.mmd_scale *  self.mmd # 在 -ELBO_gau 后面加上 mmd系数 × mmd函数
        
        else: # 如果不使用MMD，则输出 -ELBO_gau 作为损失函数， 注意有负号 ###############
            self.loss = - self.ELBO_gau
        ## Loss function 损失函数！！加上了 scvis 的tnse目标函数
        
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        optimizer = self.optimize_algo # 优化器
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.minimize(self.loss) # 训练！
    
    @define_scope
    def optimize_test(self):
        # Test time optimizer to compare log-likelihood score of ZINB-WaVE
        update_ops_test = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, "variational")
        test_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "variational")
        optimizer_test = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, epsilon=0.1)
        with tf.control_dependencies(update_ops_test):
            self.test_step = optimizer_test.minimize(self.loss, var_list=test_vars)
    
    @define_scope
    def imputation(self):
        # more information of zero probabilities
        if self.zi:
            self.zero_prob = tf.nn.softplus(- self.px_dropout + tf.exp(self.px_r) * self.px_r - tf.exp(self.px_r) \
                             * tf.math.log(tf.exp(self.px_r) + self.px_rate + 1e-8)) \
                             - tf.nn.softplus( - self.px_dropout)
            self.dropout_prob = - tf.nn.softplus( - self.px_dropout)

