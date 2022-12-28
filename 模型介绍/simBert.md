
#  <div align='center' ><font size='70'> SimBERT详解</font></div>

### 概念

SimBERT是基于微软的UniLM中的seq2seq部分设计了融检索与生成任务于一体的模型，它具备相似问生成和相似句检索能力。SimBERT是有监督训练，训练语料是自行收集到的相似句对。
SimBERT=BERT+UniLM+对比学习。

### 模型结构

SimBERT主要是有两个部分构成：
+ 第一块是构建Seq2Seq任务，也就是通过输入文本去预测对应相似文本；
+ 第二块是构建语义相似度任务，会根据文本对应的CLS向量来计算相似度。
因为有seq2seq任务生成相似度文本部分，所以simbert具备了文本相似度；因为语义相似度部分，具备了文本检索能力。
UniLM的核心是通过特殊的Attention Mask来赋予模型具有Seq2Seq的能力。假如输入是“你想吃啥”，目标句子是“白切鸡”，那UNILM将这两个句子拼成一个：[CLS] 你 想 吃 啥 [SEP] 白 切 鸡 [SEP]，然后接如图的Attention Mask：
<div align='center' ><img width="500" height="400" alt="1852906-20220623161931007-1534771153" src="https://user-images.githubusercontent.com/66345340/209788374-c4fff1e3-9d36-4f0e-99c0-91805e58e400.png"></div>

换句话说，**[CLS] 你 想 吃 啥 [SEP]** 这几个token之间是双向的Attention，而**白 切 鸡 [SEP]** 这几个token则是单向Attention，从而允许递归地预测 **白 切 鸡 [SEP]** 这几个token，所以它具备文本生成能力。
<div align='center' ><img width="600" height="200" alt="1852906-20220623161931007-1534771153" src="https://user-images.githubusercontent.com/66345340/209788511-1b3cd036-07fb-4c53-acf1-c6a1416f919c.png"></div>
UNILM做Seq2Seq模型图示。输入部分内部可做双向Attention，输出部分只做单向Attention。

Seq2Seq只能说明UniLM具有NLG的能力，那前面为什么说它同时具备**NLU和NLG** 能力呢？因为UniLM特殊的Attention Mask，所以 **[CLS] 你 想 吃 啥 [SEP]** 这6个token只在它们之间相互做Attention，而跟**白 切 鸡 [SEP]** 完全没关系，这就意味着，尽管后面拼接**白 切 鸡 [SEP]** ，但这不会影响到前6个编码向量。再说明白一点，那就是前6个编码向量等价于只有 **[CLS] 你 想 吃 啥 [SEP]** 时的编码结果，如果 **[CLS]的向量代表着句向量** ，那么它就是**你 想 吃 啥** 的句向量，**而不是加上白 切 鸡后的句向量**。

由于这个特性，UniLM在输入的时候也随机加入一些[MASK]，这样输入部分就可以做MLM任务，输出部分就可以做Seq2Seq任务，MLM增强了NLU能力，而Seq2Seq增强了NLG能力，一举两得。

#### 训练策略

+ SimBERT属于有监督训练，训练语料是自行收集到的相似句对，通过一句来预测另一句的相似句生成任务来构建Seq2Seq部分，然后前面也提到过 **[CLS]** 的向量事实上就代表着输入的**句向量**，所以可以同时用它来训练一个检索任务，下面是SimBERT训练数据格式：
![image](https://user-images.githubusercontent.com/66345340/209787192-05d66b7e-e554-4882-9073-f77f2023526c.png)

从上图可以看出，训练数据是相似文本对的方式，包括一条文本和对应的相似语义文本列表。

假设SENT_a和SENT_b是一组相似句，那么在同一个batch中，把[CLS] SENT_a [SEP] SENT_b [SEP]和[CLS] SENT_b [SEP] SENT_a [SEP]都加入训练，做一个相似句的生成任务，这是Seq2Seq部分，也就是图中右边部分。

另一方面，把整个batch内的[CLS]向量都拿出来，得到一个句向量矩阵 **V∈Rb×d（b是batch_size，d是hidden_size）**，然后对  **d** 维度做  **L2 归一化**，得到 **V~**，然后两两做内积，得到  **b×b** 的相似度矩阵 **V~V~⊤**，接着乘以一个scale（我们取了30），并mask掉对角线部分，最后每一行进行 **softmax，作为一个分类任务训练**，每个样本的目标标签是它的相似句（至于自身已经被mask掉）。说白了，**就是把batch内所有的非相似样本都当作负样本，借助softmax来增加相似样本的相似度，降低其余样本的相似度**。

**说到底，关键就是“[CLS]的向量事实上就代表着输入的句向量”，所以可以用它来做一些NLU相关的事情。最后的loss是Seq2Seq和相似句分类两部分loss之和。**

### 损失函数
**loss是Seq2Seq和相似句分类两部分loss之和。


### 对比、优缺点


### 常见的面试问题
 1、
