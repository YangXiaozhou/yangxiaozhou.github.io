---
layout: post
title:  "Francis Galton: 维多利亚时代的博学家与他观察到的奇妙世界"
date:   2020-05-17 08:00:00 +0800
categories: DATA
tags:  
---

周末读Aeon的一篇文章：[Algorithms associating appearance and criminality have a dark past](https://aeon.co/ideas/algorithms-associating-appearance-and-criminality-have-a-dark-past?utm_source=Aeon+Newsletter&utm_campaign=f7c118f081-EMAIL_CAMPAIGN_2020_05_11_01_52&utm_medium=email&utm_term=0_411a82e59d-f7c118f081-69607277)，讲现在有研究人员用机器学习算法通过人脸来判断某人犯罪的几率。文中讲到这种从人外表提取预见性特征的尝试，在犯罪学历史上并不新奇，19世纪的意大利犯罪学家Cesare Lombroso认为罪犯的脸部有独特的样貌：突出的前额、鹰型鼻梁；而18世纪的Francis Galton则尝试回答一个更广泛的问题：人的外表跟他或她的健康状况、犯罪倾向、智力等等有关系吗？或者说，人的基因是否决定了健康、行为、智力和竞争力？

### Francis Galton是谁？
这名字看起来有点眼熟，我隐约记得在老板的一门Forecasting统计课上听到过。仔细一想，对，在线性回归的部分，老板上课专门介绍了他。Sir Francis Galton，姓Galton，名Francis，但当提到他时，出于礼仪，你得加个Sir，因为他在1909年被英国女王授予了骑士爵位。为什么在讲线性回归的时候要介绍他呢？因为他作为第一个人，观察并记录了这样一种现象[^Galton_heights]：平均身高很高的父母，往往会有身高更接近普通的孩子；而平均身高偏低的父母的孩子，成年后通常有着更接近普通人的身高。
下图是[Bradley Efron](https://www.ams.org/journals/bull/2013-50-01/S0273-0979-2012-01374-5/S0273-0979-2012-01374-5.pdf)根据Galton当时收集到的父母和孩子的身高数据重新制的图，完美地展现了我们现在所知道的Bivariate normal distribution。
![regression_to_mean](/assets/francis-galton/regression_to_mean.png)

他把这种现象称为[regression towards mediocrity](https://www.jstor.org/stable/2841583)，现在通常叫做regression toward the mean，中文貌似叫“向均数回归”。同样的现象，我们在生活中很多地方都能观察到：因为运气而押中题目的学生考出了高分，下一次考试的成绩却没那么突出；连续投中三个三分球的朋友，下个球往往“容易”失手；我上周做[油泼猪手](https://yangxiaozhou.github.io/learning/2019/01/01/recipe.html#%E6%B2%B9%E6%B3%BC%E7%8C%AA%E6%89%8B)时各种调料拿捏得很好，味道超棒，这周再做一次，大概率味道会比较普通🤷‍♂️。

符合这原则的现象，他们有一个共通点：他们的结果往往完全或部分由随机因素决定，而随机因素的影响往往符合以0为中心的正态分布（时好时坏）。比如说，三分球进或不进，有投手能力的影响但也有运气的成分；我做的某道菜的味道，取决于下厨能力，但我的专心程度、手抖程度以及心情等几乎随机的因素也会有所影响。也就是说，假设某一天我超级走运，做出了迄今为止最好吃的一道菜，这种事件发生的概率是很小的（得到正态分布上的极大值或极小值的概率）。下一次做，大概率我会正常发挥，菜的味道也没上次好（取到了正态分布上0周围的某个值）。

想象这样一种情况：朋友在我搬新家的时候来家里吃饭，刚好碰到我前面说的超常发挥，都说做的猪手好吃！过了几个月，家里聚会，应朋友强烈要求，再次做出一盘猪手，不过这次是正常发挥。朋友吃后回忆起之前，评论到：“水平下降了呀！” 我冤不冤？ 这样的冤枉我们生活中还真不少，以至于它有个专门的称呼：Regression fallacy，中文叫“回归谬误”。Daniel Kahneman讲过亲身经历的这样[一个例子](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3292229/)：他有一次给飞行员学校做培训，提到了表扬能使学员变得更优秀。下面的一个教官不同意了，说他每次一夸完降落做得简直完美的学员，下一次一定做得没那么好，而刚被他骂过的学员，马上就能看到提升。听了教官的抗议，Kahneman当下有了一个eureka moment，他说道：
> because we tend to reward others when they do well and punish them when they do badly, and because there is regression to the mean, it is part of the human condition that we are statistically punished for rewarding others and rewarded for punishing them.

回归谬误可以用数学证明，假设两个变量以bivariate normal distribution分布，只要他们的correlation小于1，就会有回归谬误出现，对证明感兴趣的朋友可以看[维基](https://en.wikipedia.org/wiki/Regression_toward_the_mean)。

### 年度公牛体重竞猜
让我们回到观察小天才Galton。1907年三月的自然杂志上刊登了他一篇篇幅只有一页的[来信](https://www.nature.com/articles/075450a0)，名为：Vox Populi，直译为“民众的声音”，现在指大多数人的意见。住在英国Plymouth的他，注意到了家附近的镇子上每年都有举办这样一种家禽体重竞猜活动：主办方拉一头牛出来，参与竞猜的本地农夫、屠夫等感兴趣且有经验者对牛进行评估，并将他认为这头牛被宰杀洗净之后的体重提交上去。本着对大众智慧的科学研究态度，他通过某种方式获得了一次竞猜比赛中的数据：牛的真实体重以及787个竞猜者的估计。

![stock_show](/assets/francis-galton/stock_show.jpg)
{:style="display: block; margin-left: auto; margin-right: auto; width: 100%;"}

他把提交的所有竞猜体重从小到大排列开，发现中位数（一半的数比它低，一半比它高，"median"一词就是他给取的）是1207磅，而那头牛的真实净体重是1198磅，也就是说，民众的判断在这里跟真实值只差了0.8%！[^correction]
在那个线性回归还不是所有数据分析课程的第一节课，数据科学也还不是一种职业的时候，Galton从787个竞猜体重中通过简单的手算看到了以平均值或中位数对真实值进行估计的准确性。现在我们知道了，sample mean is an unbiased estimator of the true population mean。

Galton的观察没有停在估计的准确性上，他还想知道，每个人估计的误差有多大。他随即把所有估计与中位数的偏差画了出来，他发现，每一个有经验的“肉眼测体重者”所做的估计，从低估的到高估的，一系列的偏差与正态分布极为相似。也就是说，如果把真实净体重看做是这个采样分布的mean，那任意一个参赛者（有经验的）来估计，他的估计值将是以真实净体重为中心的正态分布而分布着的（绕口！）。Let's try again. The estimate by any pair of trained eyes is distributed normally around the true dressed weight of the ox. 这里我们得提一个无数现代科学依赖的理论：Central limit theorem (CLT)。对，就是那个可以解释为什么正态分布在现实生活中如此普遍的理论。因为CLT，我们现在确切地知道，当样本量足够大时，样本平均值呈以真实值为中心的正态分布。所以，从年度公牛体重竞猜的真实数据上，他，Sir Francis Galton，看到了central limit theorem。

附上他原稿里的跟理论正态分布做对比的图，横轴是百分位，纵轴是偏差：
![francis-galton-the-wisdom-of-crowds](/assets/francis-galton/francis-galton-the-wisdom-of-crowds.jpg)

### Galton Board与柏青哥
Galton对于这种没有征兆但又近乎定律般呈正态分布的偏差非常着迷，他把偏差呈现出来的图称为:The Curve of Frenquency，也就是我们现在熟知的样本偏差的正态分布图。为了展现这种偏差的正态分布（即，CLT)以及前面提到的回归谬误，Galton设计了一个令人拍案叫绝的装置：Galton Board，现在也叫bean machine，如下图：

![GaltonBoard](/assets/francis-galton/GaltonBoard.png)
{:style="display: block; margin-left: auto; margin-right: auto; width: 75%;"}

像不像我们小时候在街巷的小卖部里玩过的弹珠机？没错，他们的背后是同样的原理。实际上，风靡全日本的柏青哥也是用的这样的设计原理：弹珠从顶部落下，经过跟若干层的撞针的撞击，最终掉进最下面从左到右N个桶当中的某个桶里。因为我们并不知道弹珠在跟每一根撞针撞击之后是走左还是走右，所以某个弹珠的最终位置并不能提前知道（i.e. 随机事件，随机漫步，随机过程）。

但！虽然单独一个弹珠的去向无法提前获知，但我们却有办法知道某个弹珠落入某个区间的概率。粗略来说，弹珠到达某一个桶的路线数量除以所有它可能走的路线，就是它进入某个桶的概率。比如，一颗弹珠想要到达最左边的区间，它只有一条路可以走：从第一层开始一直往左弹。算出其他区间的路线数和概率可以有很多方法，比如枚举（费劲）或用斐波那契数列（你也很能观察！），也可以根据Binomial distribution的probability mass function (pmf)得到（$n$是撞针的层数，$k$是桶的编号，$p$是弹珠撞击后弹左的概率）：

$$
\operatorname{Pr}(X=k)=\left(\begin{array}{l}
n \\
k
\end{array}\right) p^{k}(1-p)^{n-k}
$$ 

for $k = 0, \dots, n$.

读到这里，了解CLT的朋友或许已经明白为什么这个Galton board可以展示呈正态分布的偏差了。CLT的一个特殊应用是证明当试验的次数($n$)足够大的时候，binomial distribution的pmf会跟正态分布十分相似。换句话说，当我们的Galton board足够大，同时扔下的弹珠足够多的时候，我们应该就能看到经典的正态分布Bell curve！Genius!

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/jiWt77xme64" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>

---------
为什么我们说这个弹珠机也能展示之前提到的回归谬误呢？首先，让我们把刚才那几百个弹珠落下来之后呈现出来的分布记在脑海中。让我再次使用做菜的例子，假设落到最右端的弹珠代表着我做出了迄今为止最好吃的一道菜，因为不寻常地走运（弹珠掉入最右边的几率非常小）。现在，我把这颗弹珠拿出来，让它从顶部再一次下落（再做一次同样的菜），你觉得大概率会是掉在哪片地方？有多大概率再次到达最右端（做出同样高水平的菜）？

呵，Life!

对于众多弹珠看似随机、无法预测地落下，最后被某种魔力聚拢，一个挨着一个，逐渐呈现出美丽的正态分布的现象，Galton自己是这样描述的：
> Order in Apparent Chaos: I know of scarcely anything so apt to impress the imagination as the wonderful form of cosmic order expressed by the Law of Frequency of Error. The law would have been personified by the Greeks and deified, if they had known of it. It reigns with serenity and in complete self-effacement amidst the wildest confusion. The huger the mob, and the greater the apparent anarchy, the more perfect is its sway. It is the supreme law of Unreason. Whenever a large sample of chaotic elements are taken in hand and marshalled in the order of their magnitude, an unsuspected and most beautiful form of regularity proves to have been latent all along.

### 结语
Francis Galton作为英国维多利亚时期的一位博学家，经历实在是太过丰富。自幼出生在富足精英的家庭，他是达尔文的表弟，年轻时继承了父亲的大笔遗产之后去非洲大陆探险，回国之后写成的游记成了畅销书。用他敏锐的观察力和好奇心，Galton研究了很多问题，有些没啥实际影响（最佳切蛋糕法、最佳沏茶法），有些却改变了众多领域接下来一百多年的发展。他做了早期的回归分析、提出了correlation的概念、将统计应用到遗传学、心理学，数理统计最重要的学者之一Karl Pearson是他的学生。同时，他为了得到数据，发明了问卷调查；研究天气，发明了第一张天气地图、开启了对气候的科学研究；提出了一种有效识别指纹的方法，对当时的法医学做了推动。哦，对了，正如我们开头所说，他也提出了一种根据不同人脸图像提取“平均特征”的方法。

Galton所观察到的世界，让他有了很多疑问，他尝试用各种方法去丈量这个世界，并从看似混沌无序的现象中找到秩序和规律。我惊叹于Galton的观察力、跟随自己好奇心不断的探索与尝试以及对自己专业不设限的态度。文艺复兴人的精神劲儿可见一斑。好了，不多说了，我要去入手一个Galton board了。

最后附上一个把Galton board解释得比我清楚得多、诙谐又幽默的哥们的[视频](https://www.youtube.com/embed/UCmPmkHqHXk)。

### 注释
[^Galton_heights]:这里放上Galton自己制作的父母孩子身高回归图：![Galton_heights](/assets/francis-galton/Galton's_correlation_diagram_1875.jpg)

[^correction]:后来的研究修正了Galton原稿的数据错误，当时那头牛的真实净体重应该是1197，而中位数估计应该是1208。在原稿中，Galton用中位数进行了真实值估计。不过，当时787个估计的平均数是1197。也就是说，平均数其实以零误差的表现估计到了真实值！
