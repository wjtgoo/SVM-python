# 支持向量机SVM（包括线性核、多项式核、高斯核）python手写实现

## 理论

参考[《统计学习方法》Chapter.7 支持向量机（SVM）](https://taotaoiit.blog.csdn.net/article/details/127952840?spm=1001.2014.3001.5502)

## 代码构架说明（SVM类）

> 借鉴sklearn的代码构架，整体功能实现在SVM类中，包括各种类属性，以及常用的模型训练函数`SVM.fit(x,y,iterations)`，以及预测函数`SVM.predict(x)`

### 类输入参数

```python
class SVM(kernal='linear', C=1)
```

`kernal:` 默认：线性核,可选：线性核('linear')，多项式核('poly')，高斯核('gauss')

`C:`惩罚参数

### 类属性

```python
def __init__(self,kernal='linear',C=1):
    # 初始化核函数
    self.kernal = kernal
    # 初始化对偶问题参数
    self.alpha = None
    self.C = C
    # 初始化决策函数参数
    self.b = np.random.rand()
    self.x = None
    self.y = None
```

**说明**：

最终决策函数由于在核为非线性核时，无法显式的表示决策函数的权重`w`，因此需要保存训练的数据`self.x`和`self.y`，所以此代码的一个缺点就是，每次预测时都需要重新对训练数据集进行运算，在数据集较大的情况下会导致SVM预测效率比较低，可能有几个tricks可以对此进行一些小优化（比如将计算的中间结果保存啥的）但是笔者有点懒，不想对此进行优化（主要是感觉治标不治本，其实我很好奇sklearn是如何应对这个问题的，但是懒惰战胜了我的好奇心）

### 类方法

#### 1 SVM.fit(...)

**说明:**

额，没啥好说的，看着挺长，其实就是SMO（序列最小优化算法）的实现，SMO其实整体思路挺简单的，由于原始的SVM的对偶问题是一个有n个变量的二次规划问题，对于n比较大的时候，很多二次规划的求解方法都比较低效，这个SMO就是专门为这种情况而生的，它从n个变量中通过一个法则，选出两个变量来优化（其他变量当作常数），又由于对偶问题有一个等式约束，正好可以把两个变量变成一个变量（一个变量有另一个变量表示）最后就能变成一个单变量的极值问题，显然通过求偏导等于零就能求出解析解，所以就能对这个问题，仅通过两个变量进行一次小小的优化，然后剩下的就是轮着换变量就行了，优化结束的标志就是变量$\alpha$满足KKT条件。

上述思路的每一步，在代码注释应该都挺清楚的

```python
    def fit(self,x,y,iterations=1000):
        '''
        训练模型
        参数说明：   x n*features_n  np.array
                    y n-vector      np.array
        USing SMO algorithm
        '''
        # SMO
        # 初始化
        self.x = x
        self.y = y
        self.alpha = np.random.rand(len(x))*self.C
        alpha1 = None
        alpha2 = None
        alpha1_id = None
        alpha2_id = None
        E_all = np.array([self.E(x,y,x[i],y[i]) for i in range(len(y))])
        while(iterations):
            # 选择第一个变量
            flag = False
            for i in range(len(x)):
                if self.alpha[i]<self.C and self.alpha[i]>0 :
                    if y[i]*self.g(x,y,x[i]) != 1:
                        alpha1 = self.alpha[i]
                        alpha1_id = i
                        flag = True
                        break
            if flag == False:
                for i in range(len(x)):
                    if self.alpha[i]==0:
                        if y[i]*self.g(x,y,x[i]) < 1:
                            alpha1 = self.alpha[i]
                            alpha1_id = i
                            flag = True
                            break
                    elif self.alpha[i]==self.C:
                        if y[i]*self.g(x,y,x[i]) > 1:
                            alpha1 = self.alpha[i]
                            alpha1_id = i
                            flag = True
                            break
            # 遍历完数据集后如果还没有发现违反KKT条件的样本点,则得到最优解
            if flag == False:
                print("get optimal alpha")
                break
            # 选择第二个变量
            E_1 = E_all[alpha1_id]
            if E_1 >=0:
                alpha2_id = np.argmin(E_all)
                alpha2 = self.alpha[alpha2_id]
            else:
                alpha2_id = np.argmax(E_all)
                alpha2 = self.alpha[alpha2_id]
            E_2 = E_all[alpha2_id]
            # 对alpha1 alpha2进行优化
            # 这里是解析解
            # 求alpha2的取值边界
            if y[alpha2_id] != y[alpha1_id]:
                L = np.max([0,alpha2-alpha1])
                H = np.min([self.C,self.C+alpha2-alpha1])
            else:
                L = np.max([0,alpha2+alpha1-self.C])
                H = np.min([self.C,alpha2+alpha1])
            # eta = K11+K22-K12
            eta = self.kernal_(x[alpha1_id],x[alpha1_id])+\
                  self.kernal_(x[alpha2_id],x[alpha2_id])-\
                  2*self.kernal_(x[alpha1_id],x[alpha2_id])
            alpha2_uncut = alpha2 + y[alpha2_id]*(E_1-E_2)/eta
            if alpha2_uncut>H:
                alpha2 = H
            elif alpha2_uncut>=L and alpha2_uncut<=H:
                alpha2 = alpha2_uncut
            else:
                alpha2 = L
            # 更新alpha
            alpha1_old = self.alpha[alpha1_id]
            alpha2_old = self.alpha[alpha2_id]
            self.alpha[alpha1_id] = alpha1+y[alpha1_id]*y[alpha2_id]*(alpha2_old-alpha2)
            self.alpha[alpha2_id] = alpha2
            # 更新 b
            b1 = -E_1 - \
                y[alpha1_id]*self.kernal_(x[alpha1_id],x[alpha1_id])*(self.alpha[alpha1_id]-alpha1_old)-\
                y[alpha2_id]*self.kernal_(x[alpha2_id],x[alpha1_id])*(self.alpha[alpha2_id]-alpha2_old)+self.b
            b2 = -E_2 - \
                y[alpha1_id]*self.kernal_(x[alpha1_id],x[alpha2_id])*(self.alpha[alpha1_id]-alpha1_old)-\
                y[alpha2_id]*self.kernal_(x[alpha2_id],x[alpha2_id])*(self.alpha[alpha2_id]-alpha2_old)+self.b
            self.b = (b1+b2)/2
            # 更新E
            E_all = np.array([self.E(x,y,x[i],y[i]) for i in range(len(y))])
            iterations-=1
```

#### 2 SVM.predict(...)

```python
def predict(self,x):
    '''
        预测函数
        输入:       x n*features_n np.array
        输出:       y_pre n-vector
        '''
    y_pre = np.zeros(len(x))
    for i in range(len(x)):
        y_pre[i] = np.sum(self.alpha*self.y*self.kernal_(self.x,x[i]),axis=-1)
        y_pre +=self.b
        y_pre = self.sign(y_pre)
```

**说明**：很正常的预测函数，对应$f(x) = {\rm sign}(\sum_{i=1}^{N}{\alpha_i^*y_iK(x_i,x)}+b^*)$，只不过这里的决策函数因为刚才说了，权重$w$不能显式表示，我用for循环加起来了（我其实试过numpy的数组计算，可惜对高斯核不太友好（怨我高斯核设计的不行，懒得改了），不然可以写的很简洁）

#### 3 SVM.kernal_(...)

```python
def kernal_(self,x,z,p=2,sigma=1):
       if self.kernal == 'linear':
           return x@z
       elif self.kernal == 'poly':
           return np.power(x@z+1,p)# 默认p=2
       elif self.kernal == 'gauss':
           return np.exp(-np.linalg.norm(x-z,axis=-1)**2/(2*sigma))# 默认sigma=1
       else:
           raise Exception("核函数定义错误！！")
```

**说明**：内部用于进行核函数选择以及计算的，看看就行，这里可以改核的参数

#### 4 SVM.g(...)

```python
def g(self,x,y,xi):
        return np.sum(self.alpha*y*self.kernal_(x,xi))+self.b
```



**说明**:对应SMO里边的函数$g(x)=\sum_{i=1}^{N}{\alpha_iy_iK(x_i,x)}+b$，不会的查《统计学习方法》P145，我博客里没写，因为太费劲了

#### 5 SVM.E(...)

```python
def E(self,x,y,xi,yi):
        return self.g(x,y,xi)-yi
```

**说明**:对应SMO里边的函数$E(x)=g(x_i)-y_i$，不会的查《统计学习方法》P145

#### 6 SVM.sign(...)

```python
def sign(self,x):
        if type(x) == np.ndarray:
            x[x>=0] = 1
            x[x<0] = 0
        elif (type(x) == float)|(type(x) == int):
            if x>=0:
                x = 1
            else:
                x = 0
        return x
```

**说明**: 就是以下函数，只不过，为了适用于更多数据类型，我加了一些判断能够对数组进行向量操作
$$
{\rm sign}(x)=
\begin{cases}
1& x\geq0\\
0& x<0
\end{cases}
$$

### 完整SVM类代码

```python
# Date:2022.11.20 20:20
# Auther: WJT
# SVM class
import numpy as np

class SVM:
    def __init__(self,kernal,C):
        # 初始化核函数
        self.kernal = kernal
        # 初始化对偶问题参数
        self.alpha = None
        self.C = C
        # 初始化决策函数参数
        self.b = np.random.rand()
        self.x = None
        self.y = None
    def fit(self,x,y,iterations=1000):
        '''
        训练模型
        参数说明：   x n*features_n  np.array
                    y n-vector      np.array
        USing SMO algorithm
        '''
        # SMO
        # 初始化
        self.x = x
        self.y = y
        self.alpha = np.random.rand(len(x))*self.C
        alpha1 = None
        alpha2 = None
        alpha1_id = None
        alpha2_id = None
        E_all = np.array([self.E(x,y,x[i],y[i]) for i in range(len(y))])
        while(iterations):
            # 选择第一个变量
            flag = False
            for i in range(len(x)):
                if self.alpha[i]<self.C and self.alpha[i]>0 :
                    if y[i]*self.g(x,y,x[i]) != 1:
                        alpha1 = self.alpha[i]
                        alpha1_id = i
                        flag = True
                        break
            if flag == False:
                for i in range(len(x)):
                    if self.alpha[i]==0:
                        if y[i]*self.g(x,y,x[i]) < 1:
                            alpha1 = self.alpha[i]
                            alpha1_id = i
                            flag = True
                            break
                    elif self.alpha[i]==self.C:
                        if y[i]*self.g(x,y,x[i]) > 1:
                            alpha1 = self.alpha[i]
                            alpha1_id = i
                            flag = True
                            break
            # 遍历完数据集后如果还没有发现违反KKT条件的样本点,则得到最优解
            if flag == False:
                print("get optimal alpha")
                break
            # 选择第二个变量
            E_1 = E_all[alpha1_id]
            if E_1 >=0:
                alpha2_id = np.argmin(E_all)
                alpha2 = self.alpha[alpha2_id]
            else:
                alpha2_id = np.argmax(E_all)
                alpha2 = self.alpha[alpha2_id]
            E_2 = E_all[alpha2_id]
            # 对alpha1 alpha2进行优化
            # 这里是解析解
            # 求alpha2的取值边界
            if y[alpha2_id] != y[alpha1_id]:
                L = np.max([0,alpha2-alpha1])
                H = np.min([self.C,self.C+alpha2-alpha1])
            else:
                L = np.max([0,alpha2+alpha1-self.C])
                H = np.min([self.C,alpha2+alpha1])
            # eta = K11+K22-K12
            eta = self.kernal_(x[alpha1_id],x[alpha1_id])+\
                  self.kernal_(x[alpha2_id],x[alpha2_id])-\
                  2*self.kernal_(x[alpha1_id],x[alpha2_id])
            alpha2_uncut = alpha2 + y[alpha2_id]*(E_1-E_2)/eta
            if alpha2_uncut>H:
                alpha2 = H
            elif alpha2_uncut>=L and alpha2_uncut<=H:
                alpha2 = alpha2_uncut
            else:
                alpha2 = L
            # 更新alpha
            alpha1_old = self.alpha[alpha1_id]
            alpha2_old = self.alpha[alpha2_id]
            self.alpha[alpha1_id] = alpha1+y[alpha1_id]*y[alpha2_id]*(alpha2_old-alpha2)
            self.alpha[alpha2_id] = alpha2
            # 更新 b
            b1 = -E_1 - \
                y[alpha1_id]*self.kernal_(x[alpha1_id],x[alpha1_id])*(self.alpha[alpha1_id]-alpha1_old)-\
                y[alpha2_id]*self.kernal_(x[alpha2_id],x[alpha1_id])*(self.alpha[alpha2_id]-alpha2_old)+self.b
            b2 = -E_2 - \
                y[alpha1_id]*self.kernal_(x[alpha1_id],x[alpha2_id])*(self.alpha[alpha1_id]-alpha1_old)-\
                y[alpha2_id]*self.kernal_(x[alpha2_id],x[alpha2_id])*(self.alpha[alpha2_id]-alpha2_old)+self.b
            self.b = (b1+b2)/2
            # 更新E
            E_all = np.array([self.E(x,y,x[i],y[i]) for i in range(len(y))])
            iterations-=1
    def predict(self,x):
        '''
        预测函数
        输入:       x n*features_n np.array
        输出:       y_pre n-vector
        '''
        y_pre = np.zeros(len(x))
        for i in range(len(x)):
            y_pre[i] = np.sum(self.alpha*self.y*self.kernal_(self.x,x[i]),axis=-1)
        y_pre +=self.b
        y_pre = self.sign(y_pre)
        return y_pre
    def kernal_(self,x,z,p=2,sigma=1):
        if self.kernal == 'linear':
            return x@z
        elif self.kernal == 'poly':
            return np.power(x@z+1,p)# 默认p=2
        elif self.kernal == 'gauss':
            return np.exp(-np.linalg.norm(x-z,axis=-1)**2/(2*sigma))# 默认sigma=1
        else:
            raise Exception("核函数定义错误！！")
    def g(self,x,y,xi):
        return np.sum(self.alpha*y*self.kernal_(x,xi))+self.b
    def E(self,x,y,xi,yi):
        return self.g(x,y,xi)-yi
    def sign(self,x):
        if type(x) == np.ndarray:
            x[x>=0] = 1
            x[x<0] = 0
        elif (type(x) == float)|(type(x) == int):
            if x>=0:
                x = 1
            else:
                x = 0
        return x
```

## 测试

> 总的来说，我个人对这个效果挺满意的，就是高斯核情况下有点慢，然后高斯核居然有的时候没有线性核好？我也不知道为啥，但有一点需要指出，以下的数据只有100来个，精度低精度高全凭`train_test_split`的心情。

##### 调用鸢尾花数据集
SVM是二分类模型，由于原来的鸢尾花数据集是3类，第三类的数据全部剔除，只留下两类数据，并划分训练集与测试集。
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
Y = iris.target
X = X[Y!=2]
Y = Y[Y!=2]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)
```
##### 测试
构建类实例并训练
```python
svm_clf = SVM('gauss',C=1)
svm_clf.fit(X_train,Y_train,iterations=1000)
```
预测

```python
y_pre = svm_clf.predict(X_test)
print('验证集正确率：',(y_pre == Y_test).sum()/len(Y_test))
```

```python
验证集正确率： 1.0
```