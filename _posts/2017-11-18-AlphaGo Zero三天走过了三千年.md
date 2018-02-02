---
layout: post
title: AlphaGo Zero三天走过了三千年
tags: [深度学习]
excerpt: "人工智能这一两年发展迅速，成为大家热捧的一个领域，这两天波斯顿机器人ATLAS的后空翻简直帅爆，让人惊叹至极，此文针对10月19日Google人工智能团队在Nature发表的《[Mastering the game of Go without human knowledge][1]》做一个简要的解读"

---

 1. AlphaGo Zero介绍
 * 什么是AlphaGo
 * 为什么选择围棋
 * 围棋规则概括
 2. AlphaGo Zero实现原理
 * 新旧AlphaGo的比较
 * AlphaGo Zero的实现原理
 3. 人类与人工智能
 * AlphaGo Zero的意义
 4. 问题
 
---

## 一. AlphaGo Zero介绍

**1.什么是AlphaGo**
* (1)阿尔法围棋（AlphaGo）是位于伦敦的Google DeepMind 团队开发的一款围棋人工智能程序，程序利用`价值网络`去计算局面，用`策略网络`去选择下子。
* (2)AlphaGo 的主要工作原理是`强化学习`，其算法结合了`神经网络`、`机器学习`和`蒙特卡罗树搜索`技术。AlphaGo Zero是最新版本。
* (3)Google DeepMind 团队10月19日在Nature上发表了这篇论文， AlphaGo Zero 横空出世，完全`从零开始`，不需要参考人类任何的先验知识，完全靠自己一个狗对弈强化学习（Reinforcement Learning）和参悟，棋艺增长远超阿法狗，百战百胜，击溃其他版本100-0。
* (4)达到这样一个水准，AlphaGo Zero 只需要在`4`个TPU上，花三天时间，自己左右互搏`490`万棋局。前面版本需要在48个TPU上，花几个月的时间，学习三千万棋局，才打败人类。
* (5)AlphaGo Zero 没有任何关于围棋的先验知识，仅仅输入了围棋基本规则，完全从零开始学习;3天，AlphaGo Zero 实现对AlphaGo Lee（打败李世石的AlphaGo版本）的碾压，——对弈成绩为100:0；21天，AlphaGo Zero 达到 AlphaGo Master （打败的中国棋手柯洁AlphaGo版本）的水平，在线击败了60位顶尖专业选手，并在2017年世界竞标赛上以`3:0`击败冠军柯洁。40天,AlphaGo Zero 超过其他所有 AlphaGo 版本，成为世界上最好的旗手，达到这一水平没有人类干涉，没有使用历史数据，完全依赖于自学。

> AlphaGo Zero的惊人成绩，意味着AI领域一场理论交锋的尘埃落定：`无监督学习`战胜了`监督学习`，简约而不简单，大道至简，对于AlphaGo，最简洁的，就是最美的。


<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/AlphaGo/1.GIF" style="width:450px">

---

**2.为什么选择围棋**
* 围棋棋盘为`19*19=361`格局，可能性高达`10的170`次方，围棋算法具有高度的复杂性和代表性;
* 计算机很难破解，对算法和计算能力提出了很高要求;
* 围棋可以很好地检验和测试研究水平和掌握人工智能并行计算的交互能力;
* 从围棋算法可以推广到深度学习应用的一般情形;

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/AlphaGo/2.jpg" style="width:450px">

---

**3.围棋规则简要概括**

* 19*19，一共`361`个交叉点；
* 黑白棋子依次落子,占地大者获胜；
* 上下左右相邻的,同色棋子为`整体`；
* 棋子上下左右空地为`气`,无气的子被“吃”；

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/AlphaGo/3.png" style="width:450px">

---


## 二. AlphaGo Zero 的实现原理
**1.新旧 AlphaGo 比较**

* 算法：自对弈强化学习，完全从`随机`落子开始，不用人类棋谱；
* 数据机构：只有黑子白子两种状态。之前包含这个点的气等相关棋盘信息；
* 策略：基于训练好的这个神经网，进行简单的树形搜索。之前会使用`蒙特卡洛算`法实时演算并且加权得出落子的位置；
* 模型：使用一个神经网络。之前使用了策略网络（基于深度卷积神经网）学习人类的下棋风格，局面网络来计算在当前局面下每一个不同落子的`胜率`；

>老版本的AlphaGo虽然虽然神功小城，但斧凿痕迹显著，就像机器人女友，纵有绝色容颜，但却长着机械手。Zero简洁，浑然天成，就像死宅的女神新垣结衣。

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/AlphaGo/4.png" style="width:450px">

---

**2.AlphaGo Zero的实现原理**

* AlphaGo Zero = `启发式搜索` + `强化学习` + `深度神经网络`：使用深度神经网络的训练作为`策略改善`，蒙特卡洛搜索树作为`策略评价`的`强化学习`算法。

> 策略网络和价值网络`合并`，组成一个可以同时输出`策略P`和`价值V`的新网络。

* AlphaGo Zero问题描述
1. 棋盘状态向量：s =（1,0,-1,....）
2. 落子行动：a = (0,....0,1,0,....) 
3. 落子策略：π = (0.01,0.02,0.03,...,0.93,....)
4. 损失函数：l = （z - v）^2 - π^Tlog(P) + c//θ//

> 任意给定一个状态 ⃗s，寻找最优的应对策略 ⃗a，最终可以获得棋盘上的最大地盘

* Policy Network 策略网络
<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/AlphaGo/6.jpg" style="width:450px">

* Value Network 价值网络
<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/AlphaGo/7.png" style="width:450px">

* 强化学习的通用策略迭代（Generalized Policy Iteration）方法：
1. 从策略 π0 开始；
2. 策略评估（Policy Evaluation）- 得到策略 π0 的价值 vπ0 （对于围棋问题，即这一步棋是好棋还是臭棋）；
3. 策略改善（Policy Improvement）- 根据价值 vπ0 ，优化策略为 π1； （即人类学习的过程，加强对棋局的判断能力，做出更好的判断）
4. 迭代上面的步骤2和3，直到找到最优价值 v∗，可以得到最优策略 π*；

> * 策略评估过程，即使用MCTS搜索每一次模拟的对局胜者，胜者的所有落子（⃗s）都获得更好的评估值;
> * 策略提升过程，即使用MCTS搜索返回的更好策略 π;
> * 迭代过程，即神经网络输出 p 和 v 与策略评估和策略提升返回值的对抗（即神经网络的训练过程）。
> **AlphaGo Zero 将两个网络融合成一个网络**

* 自对弈过程
【a图】表示自对弈过程s1,…,sT。在每一个位置st，使用最新的神经网络fθ 执行一次MCTS搜索αθ。根据搜索得出的概率 at∼πi 进行落子。终局 sT 时根据围棋规则计算胜者 z，πi 是每一步时执行`MCTS`搜索得出的结果（柱状图表示概率的高低）。
<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/AlphaGo/8.jpg" style="width:450px">

* 神经网络训练
【b图】表示更新神经网络参数过程。使用原始落子状态 ⃗st 作为输入，得到此棋盘状态 ⃗st下下一步所有可能落子位置的概率分布 pt 和当前状态 ⃗st下选手的赢棋评估值 vt，以最大化 pt 与 πt 相似度和最小化预测的胜者 vt 和局终胜者 z 的误差来更新神经网络参数 θ ，下一轮迭代中使用新神经网络进行自我对弈
<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/AlphaGo/9.jpg" style="width:450px">

* 强化学习

1.决策过程如下：
<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/AlphaGo/10.png" style="width:450px">
2.计算总的回报值V：
<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/AlphaGo/11.png" style="width:450px">
3.选择一个最佳的策略使得`回报`最大：
<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/AlphaGo/12.jpg" style="width:450px">

>.....** Bellman等式**

* 深度神经网络
1. 【网络输入】0/1值：现在棋盘状态的 ⃗s 以及7步历史落子记录。0白1；
2. 【网络输出】两个输出：落子概率（`362`个输出值）和一个评估值（`[-1,1]`之间；
3. 【落子概率 p】 向量表示下一步在每一个可能位置落子的概率，又称先验概率 （加上不下的选择）；
4. 【评估值 v】 表示现在准备下当前这步棋的选手在输入的这`八步`历史局面⃗s下的胜；
5. 【网络结构】基于Residual Network（ImageNet冠军`ResNet`）的卷积网络，包含`20或40`个Residual Block（残差模块），加入`批量归一化`Batch normalisation与`非线性整流器`rectifier non-linearities模块。

* 蒙特卡洛搜索树算法

1. 首先模拟一盘对决，随机面对一个空白棋盘，最初我们对棋盘一无所知，假设所有落子的方法分值都相等，设为`1`；
2. 随机 从361种方法中选一种走法 ⃗s0，在这一步后，棋盘状态变为 ⃗s1。假设对方也和自己一样随机走了一步，此时棋盘状态变为 ⃗s2；
3. 重复以上步骤直到 ⃗sn并且双方分出胜负，此时便完整的模拟完了一盘棋，我们假设一个变量r，胜利记为1，失败则为0；


<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/AlphaGo/13.jpg" style="width:450px">

---

## 三. 人类与人工智能

**1.AlphaGo Zero 的意义**
> AlphaGo Zero 的胜利是人工智能研究的一个重要里程碑。首次实现了“无监督学习”战胜了“监督学习”，简约而不简单，大道至简，九九归一，已然成了神。对于AlphaGo，最简洁的，就是最美的.

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/AlphaGo/14.jpg" style="width:450px">

---

## 四. 问题

**1.为什么阿尔法元能够完全自己学习？**
> 系统从一个对围棋一无所知的神经网络开始，将该神经网络和一个强力搜索算法结合，自我对弈。在对弈过程中，神经网络不断调整、升级，预测每一步落子和最终的胜利者。AlphaGo Zero 完全不使用人类的经验棋局和定式，只是从基本规则开始摸索，完全自发学习。

**2.AlphaGo算法能扩展到哪些领域？**
> 特性：
1.首先，它没有噪声，是能够完美重现的算法；
2.其次，围棋中的信息是完全可观测的，不像在麻将、扑克里，对手的信息观测不到；
3.最后也是最重要的一点，是围棋对局可以用计算机迅速模拟，很快地输出输赢信号。
>比如新药研发，“输赢信号”能不能很快输出，无法快速验证导致应用此处于怀疑态度。如果用数据中心节能，比较合理，因为它和围棋的特性很一致，能快速输出结果反馈，也就是AlphaGo算法依赖的弱监督信号。

**3.AlphaGo zero 是否为监督学习的争议？**
> 首先，它排除前版本的人类棋谱数据的输入训练，自我训练学习，此种方式称之为无监督学习方式。但若看棋局的精准规则的输入这个问题，则算为监督学习（南大周志华评论）

**4.AlphaGo Zero 代表「数据为王」这一标准的崩塌？**
> 围棋本身就是一种「只需要很简洁的规则就能完全描述」的问题，这样的问题理应可以用纯推理的方式解决。机器学习领域的许多其它问题，比如图像识别、语音识别、机器翻译，都不是用简洁的规则能描述得了的。在这些领域中，暂时「数据为王」这一标准仍然成立。

---
---

## 参考
* 1.[如何评价 DeepMind 发表在 Nature 上的 AlphaGo Zero？ - 知乎](https://www.zhihu.com/question/66861459?rf=66868702
)
* 2.[深入浅出看懂AlphaGo如何下棋 | Go Further | Stay Hungry, Stay Foolish](https://charlesliuyx.github.io/2017/05/27/AlphaGo%E8%BF%90%E8%A1%8C%E5%8E%9F%E7%90%86%E8%A7%A3%E6%9E%90/
)
* 3.[深入浅出看懂AlphaGo元 | Go Further | Stay Hungry, Stay Foolish](https://charlesliuyx.github.io/2017/10/18/%E6%B7%B1%E5%85%A5%E6%B5%85%E5%87%BA%E7%9C%8B%E6%87%82AlphaGo%E5%85%83/
)
* 4.[Mastering the game of Go without human knowledge : Nature : Nature Research](http://www.nature.com/nature/journal/v550/n7676/full/nature24270.html)
