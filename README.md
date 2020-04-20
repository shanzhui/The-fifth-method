# The-fifth-method
小分子成药属性预测-B榜排名第五方案
##小分子成药属性预测-B榜排名第五方案

## 0、前言
>这里我不得不吐槽一下，这个学期课太多了啊，全天满课了解一下，最终没能好好做数据分析，队伍里的其他成员也忙着其他事情，只能抽时间来做，
而且训练一次时间又比较长（这里辛苦的我的电脑了），最终采取了简单暴力又直白的方法，具体解释在后边。

## 1、源文件构成

### 1.feature.py文件
>这个文件就只是简单地提取匿名特征，这里的提取方法借鉴自初期baseline，因为本人专业不是学化学的，经过简单地前期分析无果之后，就没有提取太多其他的特征。
### 2.model.py文件
>这个文件是我们方案的核心，实际上是暴力训练很多模型，假设通过训练获得模型S1，S2.....Si.....Sn，其中n为模型数量，然后使用此n个模型的输出，作为新的特征，
训练模型D，以D的输出即可作为最终结果，但是我们在这里获得了多个模型Dj，于是之后可以做模型融合。
###3.stacking.py文件
>这个文件模型将得到的模型做集成，因为在2.feature.py中获得了多个模型Dj，于是在这个文件中将它们做模型融合。

## 2、方案思路

### 方案思想：
>本题中最主要的特征也是占比最大的特征就是三千多个匿名特征，常规建模效果并不好，那么，我们就分为两个层次：
#### 第一个层次：
>使用部分特征建立子模型，并用来预测结果，将得到的结果收集起来，有多少个子模型，就得到多少新的特征。
该步骤可以表述为：有匿名特征集合F，每次取N个特征进行建模，共进行n = int(F/N)+1步，每一步获得子模型Si，其中1<=i<=n，每个模型的输出为SOi。
当然这里我们的选取方式比较直接，选取方式是按顺序选取，比如[0,N-1],[N,2*N-1]......。
#### 第二个层次：
>使用新得到的特征取代三千多个匿名特征得到新的数据集，然后在建立一个模型作为最终输出。
该步骤可以表述为：将第一层获得的n个输出作为新的特征集合--{SOi|1<=i<=n},然后训练模型模型，获得模型Dj,其中1<=j<=m，m为使用不同的N的数量。
即，我们可以使N=300，然后训练11个子模型，然后训练获得模型D。

### 具体步骤

#### (1)数据处理，匿名特征生成
#### (2)获取子模型
>具体步骤为：

>**①选定要划分的匿名特征数量N**
>**②依次训练子模型**
>**③使用子模型做预测，将多个子模型的将结果合并输出到文件中**
>**④重新选择N，再重复上述步骤**
>**⑤xgb和lgb各自重复上述步骤**

#### (3)stacking模型融合

>经过以上步骤，可能得到多个模型，比如每次使用200个特征建立16个xgb子模型，或每次使用500个特征建立7个lgb子模型，这样第一层次的子模型的输出，
就是第二层次模型的输入，因为子模型使用的特征数不一样，第二个层次也会得到多个模型，最终使用stacking方法将m个模型进行融合。

## 3、运行说明
>运行时将Molecule_prediction_20200312文件夹与三个源文件放在同一目录下。
首先运行1.feature.py文件，生成包含匿名特征的训练集df_train和测试集df_test
然后运行2.model.py文件，使用不同数量的匿名特征进行建模，并使用lgb和xgb各进行一轮，比如，每次使用300个特征进行建模，
获得11个不同的子模型，将子模型的预测结果收集起来，视为新数据集的特征，外加4个没有用到的特征，得到共计15个特征的新数据集。
最后，将使用xgb的子模型得到的新数据集再次训练多个xgb模型，使用lgb的子模型得到的新数据集再次训练多个lgb模型，然后做stacking，得到最终的结果。

## 4、不足与反思

### 1、数据分析是硬伤。
>数据分析不够，仅使用过相关系数进行变量选择，但是，提升不大，因此在使用此方案时处理得也比较粗糙，我感觉还是有提升空间的，比如使用其他模型作为子模型。catboost也使用过，但是效果并不好。
或者N个特征也可以不是顺序选取的，可以使随机选取的，这个我没有尝试，但是我觉得可以一试。
### 2、工程能力不足。
>除了知识的欠缺，我觉得我在工程实践上，比如代码组织，文件格式与命名等很多方面都还需要继续努力。
