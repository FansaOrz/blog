---
title: SLAM算法工程师面经
date: 2023-02-10 13:08:03
tags: [秋招, 实习, SLAM, 面经]
math: true
categories: 面经
excerpt: 记录自己在2022年暑期实习和秋招过程中面试被问到的问题
---
# 暑期实习
## 字节AI Lab
### 一面
1. IMU预积分解决了什么样的问题？为什么要加入预积分？预积分有什么作用？
2. IMU预积分和普通积分有什么区别
3. FAST-LIO2有没有用到预积分
4. 比赛中用的多少线的激光雷达，有没有试过16线激光雷达+FAST-LIO2
5. FAST-LIO2和LIO-SAM前端有什么区别
6. FAST-LIO2前端如果也加上特征提取的话和LIOSAM的效果会有什么样的差别
7. LIOSAM是如何解决特征退化的
8. 你们在做室内地图构建的时候有没有遇到走廊退化的问题，你们是怎么解决的
9. 比赛时为什么要用UKF解决机器人高频振动的定位问题
{% asset_img AI_LAB_1.png AI_LAB %}

### 二面
面试官为非SLAM岗，所以没问技术细节，主要在了解项目和比赛内容

## 旷视
### 自动驾驶一面
**主要都是针对项目细节问的**
1. 介绍一下UKF、iterated-KF
2. 图优化用的什么框架，g2o和其他的框架比有什么区别
3. Eigen遇到过什么报错？有没有关于内存对齐的错误
4. 手撕：合并有序链表（[Leetcode 21题](https://leetcode.cn/problems/merge-two-sorted-lists/)）
### 研究院一面
1. 向量内积和外积的几何含义
2. UKF的状态空间有啥，除了机器人的六维状态，还有没有三维的feature
3. 手撕：给定平面中四个点，判断四个点能否形成正方形

## 图森
### 一面
1. 自我介绍、问简历里的项目
2. 紧耦合的Lidar和IMU中，IMU是如何初始化的
3. UKF的具体实现步骤
4. 定位过程中是如何处理IMU的预测值和odom的预测值
5. iterated-KF是如何工作的
### 二面
1. git如何撤销add操作
2. git如何撤销commit操作
3. fork之后，如果原仓库更新了，如何更新本地的仓库
4. C++资源管理RAII
5. C++11中的移动语义
6. 手撕1
{% asset_img tusimple_1.png tusimple_1 %}
7. 手撕2
{% asset_img tusimple_2.png tusimple_2 %}

## 达摩院
### 一面
1. 你认为在本科到研究生期间，收获最大的项目或者比赛是什么？（并介绍）
2. 你在这个项目中遇到的最大的难点是什么
3. 你做的LIO前端为什么使用IMU的预积分，预积分和普通的积分比有什么区别
4. 你提到的LOAM算法，它的帧间匹配是如何实现的
5. 你在跨越险阻这个比赛中最大的贡献是什么？（并介绍）
6. 刚才提到的NDT和ICP，介绍一下二者分别是如何实现点云匹配的
7. 一些比较基础的问题：姿态有哪几种表示形式？（四元数、欧拉角、旋转矩阵）分别有什么优缺点？欧拉角如何做插值？旋转矩阵转欧拉角是唯一解嘛？
### 二面（主管）
1. 问项目经历
2. 了解比赛内容及收获
3. 比赛准备过程中，意见不一致怎么处理的
4. 无技术细节、无笔试
### 三面（HR）
1. 你期望在这份实习中学到什么
2. 你怎么看待机器人和自动驾驶的前景
3. 以现在的技术和经验，你觉得自己会更适合去做自动驾驶还是机器人
4. 在团队中更倾向于什么角色
5. 团队中出现意见不一致的时候，你是如何解决的？举一个具体的例子
6. 说说你大学期间最tough的一段时间
7. 你有什么比别人好的学习方法嘛
8. 小蛮驴前段时间在郑州的大学中陷到水泥里面（是刚铺的水泥，还没干，只有一条细线作为标识），你会如何来分析这个问题

## 文远知行
### 一面
1. 给定p1...pn的点，用RANSAC如何拟合平面
2. 给定p1...pn的点，用最小二乘法如何拟合平面，写一下最小二乘法的方程以及如何求解
3. 给三个点，如何求平面方程
4. 说一下LIO的流程
5. LIO如何去除点云的畸变，用的pose是如何得来的
6. 使用预积分的目的
7. 写一下IMU预积分得到的 $\Delta R_{ij},\Delta v_{ij},\Delta p_{ij}$ 的形式
8. 说一下 $\Delta R_{ij},\Delta v_{ij},\Delta p_{ij}$ 的几何意义
9. 多线程之间如何线程同步（举了一个例子，前端处理好的点云数据，后端如何知道处理好了并接受）（ROS Topic？）
10. 
```c++
class A {
   static int a_;
}
```
上述代码中静态变量如何赋值
11. shared_ptr和unique_ptr的区别，以及unique_ptr如何赋值
12. 手撕：给定一个二维数组，判断其中1组成的块的个数（上下左右任意一个方向有相邻的1就组成一个块）
```c++
input:
std::vector<std::vector<bool>> grid =
{ {0, 1, 0, 0, 1},
  {0, 1, 1, 0, 1},
  {0, 0, 0, 0, 0},
  {1, 0, 0, 0, 1} };
output:
4
```

## 美团无人车
### 一面
1. 聊项目
2. IMU预积分和普通积分有什么区别
3. UKF和EKF区别
4. 手撕：输入一个浮点类型的数据，输出它的平方根（二分或牛顿法）（[LeetCode 69](https://leetcode.cn/problems/sqrtx/description/)）

### 二面
1. IMU的bias估计不准确会对预积分过程以及里程计估计有多大的影响
2. IMU预积分
3. 简要说明一下预积分的推导过程
4. 预积分的离散化是如何做的
5. IMU和Lidar的标定是如何实现的
6. 说一下LIO-Mapping、LIO-SAM、FAST-LIO2、HDL的优缺点
7. ROS的Topic这种机制有什么优点
8. 有没有接触过视觉和GPS
9. 你所做的框架有没有想过在室外环境应如何修改
10. 说明一下UKF实现定位的过程
11. C++11中的move函数
12. 
```c++
int array[4] = { 10,10,10,10 };
cout << *((double*)array) << endl; // 输出什么信息
```
13. 手撕：删除链表的倒数第N个节点（[LeetCode 19](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)）

## 元戎启行
### 一面
1. 对于平面误匹配的情况，有没有遇到过，是如何解决的（鲁棒优化的方式：①最大化共识，RANSAC实现②鲁棒核函数，M估计）
2. 预积分对于优化的意义是什么
3. 预积分的 $\Delta R_{ij},\Delta v_{ij},\Delta p_{ij}$ 有没有几何意义
4. FAST-LIO2和FAST-LIO的区别（i-KD Tree和对ikf的修改）
5. 在UKF定位中，由于UKF是针对当前时刻IMU进行预测，对于Lidar和IMU的时间不同步，是如何处理的？（可以参考一下MSCKF）

### 二面
1. C++11 智能指针
2. C++11 左值引用
3. C++11 move函数有什么作用，为什么C++11要引入move函数
4. 讲一下C++中template
5. 
```c++
/*
在 c++ unordered_map的基础上，实现可以通过timestamp支持历史记录查询的功能,
具体解释如下：
假设我们设计了这样一个history_map:
当timestamp为5的时候，插入了key=1,value=1的键值对，
如果查询key=1时，第零秒到第四秒的value，应该返回找不到；
如果查询五秒和以后的value，应该返回1。
同样的，如果在第10秒把key=1对应的value更新为2,
那么第0-4秒，key=1的value不存在，第5-9秒
key=1时value应该为1，10秒及以后，key=1的value变成2了。
请实现这样一种history_map.
*/

// 参考接口， 可按需修改

int GetCurrentTime();

class HistoryMap {
public:
    unordered_map<int, vector<pair<pair<int, int>, int>>> map; // <key,<<start,end>,value>>
    bool Get(int key, int* value) {
        if (!map.count(key)) {
            return false;
        }
        auto hisValue = map[key];
        value = hisValue[hisValue.size() - 1].second; // 返回最新时刻value
        return true;
    }

    void Set(int key, int value) {
        int currentTime = GetCurrentTime();
        if (!map.count(key)) { // 新key
            vector<pair<pair<int, int>, int>> tempValue;
            vector.push_back(pair<pair<int, int>, int>((currentTime, currentTime + 1), value)); // 未知结束时间
        }
        else {
            auto hisValue = map[key];
            hisValue[hisValue.size() - 1].first.second = currentTime; // 更新上一个结束时间
            hisValue.push_back(pair<pair<int, int>, int>((currentTime, currentTime + 1), value)); // 添加新值
        }
    }

    bool GetByTime(int key, long timestamp, int* value) {
        if (!map.count(key)) {
            return false;
        }
        for (auto eachHisValue : map[key]) {
            if (timestamp > eachHisValue.first.first && timestamp < eachHisValue.first.second) {
                value = eachHisValue.second;
                return true;
            }
        }
        auto hisValue = map[key];
        if (timestamp > hisValue[hisValue.size() - 1].first.first) { // 最后一个时间单独判断
            value = hisValue[hisValue.size() - 1].second;
            return true;
        }
        return false;
    }
};
```

### 三面
1. 自我介绍、问项目
2. 表示旋转的三种方法，各有什么优缺点
3. 求解 $ \frac{\partial R_1R_2}{\partial R_2} $

## 智加科技
### 一面
1. ROS中消息传输的机制
2. ROS中消息的时间同步
3. 项目
4. 用过什么C++中的容器（vector queue map unordered_map）
5. map和unordered_map内部分别用什么实现的
6. 模板实例化是在编译期还是在执行期
7. 智能指针有什么作用
8. 手撕：写一个快排
9. 手撕：给两组二维坐标序列，分别代表左右两条车道线，计算车道线的中线的点序（左右点的数量不一样长）

# 秋招
## 蔚来
### 一面
1. 实验室项目
2. 实习项目
3. 无专业知识无八股
4. 手撕：nums=[3,2,1,0,4]，代表每个下标可以跳跃的距离，起始位置在idx=0，问判断能不能跳跃到最后一个下标（动态规划），说一下实现的时间复杂度和空间复杂度
5. 反问：蔚来自动驾驶部门主要是在北京，目标做L4，目前地图定位组有80个人，作息10:30-21:30

### 二面
1. 实验室项目
2. 实习项目
3. 无专业知识无八股
4. 手撕：输入一个字符串，输出其计算结果，例如输入"0+3-2-5*6"， 输出-29
5. 面试官是做机器学习构建高精地图的，生成车道线、路口等

### 三面
1. 手撕：K-Means，不咋了解机制，换一个：01矩阵，找最大的相邻的1的面积
2. 九十月份能不能先过去实习
3. bfs和dfs的区别
4. fast-lio2和liosam的区别，在哪些场景下效果会更好
5. g2o和gtsam之间的区别（没用过gtsam，不太清楚）
6. fast-lio2迭代次数一般设置多大（经验来看，没改过迭代次数，不过说了一下程序判断跳出迭代的条件）
7. 意向工作地点，上海能不能接受

## 速腾聚创
### 一面
1. 聊项目，无专业知识，无八股
2. 团队规模做SLAM的五六个人，今年要扩招，往机器人公司探索，做一些自动泊车这种

### 二面
1. 聊项目，无专业知识，无八股
2. 数学题：
AB+CD+EF+GH = III，每个字母对应0-9一个数字，问哪个数字没被用到
3. 反问：
今年团队规模要扩大，再招十来个做SLAM的

### 三面
1. 聊天

## 航天科技创新研究院-HR综合面
1. 自我介绍
2. 科研结果怎样产业化，自己的看法
3. 比赛中收获最大的是什么时候
4. 为什么一直参加比赛
5. 为什么选择来研究所

## 美团无人配送
### 一面
1. 面试官是做规划控制的，不知道为啥把我捞起来了，离谱
2. 没啥好聊的，就聊项目
3. 手撕：编辑距离，没做出来（[LeetCode 72](https://leetcode.cn/problems/edit-distance/)）
4. 八股：深拷贝和浅拷贝的区别

### 定位组一面
1. 自我介绍 聊项目
2. 无八股
3. 手撕：计算根号2，循环在50次以内（直接二分）
4. 反问：定位建图规模在大几十人，具体数字不方便透露，以公开道路无人配送为主，近期不考虑做卡车

### 定位组二面
1. 项目
2. 无八股 无手撕

### 定位组三面
1. 机器人和自动驾驶，未来会更倾向于哪个方向
2. 未来的职业发展规划，选择建图或者是定位
3. 实验室做的项目、比赛或者实习期间遇到的难点，是怎么解决的
4. 三个词形容一下自己，并分别举个例子
5. 现在获取新的知识的方式有哪些
6. 现在面了哪些公司，收到了哪些意向书，你会如何排序选择
7. 会根据项目问一下项目有多少人，都是怎么分工的
8. 实验室期间有没有发论文
9. 在实验室期间主要以参加比赛为主还是做项目搞科研为主

## 擎朗智能
### 一面
1. 自我介绍
2. 项目
3. UKF实现定位
4. 运动规划里可以怎么用图优化（TEB）
5. 牛顿法、高斯牛顿法、梯度下降法
6. new和malloc区别
7. 多态
8. static关键字
9. AMCL和MCL区别、GMAPPING原理

### 二面（两个人一起面试）
1. 自我介绍（大概二十多分钟，做了一个PPT）
2. 问项目和细节
3. 滤波的五个步骤
4. 梯度下降、牛顿法、高斯牛顿法区别，梯度下降法局限性
5. 超多八股：介绍多态、介绍stl容器、map和unordered map、用过的设计模式（只知道单例模式）、智能指针、C++的内存结构

## 智加科技
### 一面
1. 项目
2. 介绍IMU预积分（好久没看了，讲的不太清楚）
3. 手撕：寻找第K大（快排）（[LeetCode 剑指Offer 76](https://leetcode.cn/problems/xx4gT2/)）
4. 开放性问题： 在做融合的时候要怎么设计各个传感器的方差
5. 如何判断lidar在三个轴上匹配的置信度
6. 点云的PCA（主成分分析）特征值和特征向量对点云来说有什么物理含义

### 二面
1. 项目
2. 问了很多达摩院这边的工作
3. C++调试工具 gdb
4. 进程和线程的区别
5. 手撕1：写一个二分（面试官很惊讶，一面居然出了寻找第K大这么难的题）
6. 手撕2：栈的压入、弹出（[LeetCode 剑指Offer 31](https://leetcode.cn/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)）
7. 反问：北京建图定位规模七八人

### 三面
1. 项目
2. 开放问题：对于高速加隧道这种场景，你可以怎么设计定位的模块
3. 开放问题：SLAM分为哪几个部分，分别是干啥的
4. 进程和线程、进程之间如何通信、ROS有没有遇到什么bug、引用传参好处、
5. 手撕：反转链表（要怎么设计验证数据 （[LeetCode 剑指Offer 24](https://leetcode.cn/problems/fan-zhuan-lian-biao-lcof/)）
6. 手撕：设计LFU（[LeetCode 460](https://leetcode.cn/problems/lfu-cache/)）

## 华为
### 一面
1. 项目
2. 手撕：给定一个字符串s，根据字符出现的频率，对其进行降序排序，例如："tree"，输出"eert"，e出现了两次，排在前面，r和t各出现一次，任意顺序都ok（共享屏幕在本地IDE做，限时15min）
3. 笔试复盘：描述一道笔试题
4. 北京上海两地办公

### 二面
1. 项目
2. 开放问题：如果机器人是坐电梯上到二楼，那么LIO会出现什么问题，该如何解决
（紧耦合会出现比较大问题，LIDAR 残差和IMU 残差不一致。解决方案：IMU零速区间检测？不太懂）
3. 开放问题：目前根据hd map、车道线定位、RTK已经能实现比较好的定位，那SLAM在无人车的定位里面主要优势是什么
（就是在没有车道线和阴天rtk信息不好的时候发挥作用？）
4. 开放问题：高精地图在自动驾驶里面比较大的问题在哪？可以如何解决？
（高精地图体量太大，可以把高精地图分块，每次只加载附近区域的地图块；可以对高精地图进行压缩，提前进行地图压缩，加快地图加载速度）
5. 手撕：01矩阵寻找岛屿数量

### 三面
1. 聊天

## 仙途
### 一面
1. 聊项目
2. 八股：常用的STL容器
3. vector的内存管理，是如何做到动态分配的
4. deque如何实现前端和后端插入删除复杂度都是O1
5. unordered map底层是用什么实现的
6. 如果要实现一个哈希表，要怎么做
7. 常用的优化算法
8. 如果图优化中有10个顶点，3条平面边，一条里程计边，那算出来的雅可比矩阵是什么样的
9. 手撕：给一个排好序的数组，把他转换成任意无序的状态

## 百度
### 一面
1. 项目
2. 无八股
3. 旋转有几种表示方法
4. 如何进行四元数插值
5. 描述点到点ICP算法过程
6. 手撕：寻找两个正序数组的中位数（Hard）（[Leetcode 4](https://leetcode.cn/problems/median-of-two-sorted-arrays/)）

反问：计算机3D视觉就是高精地图，整个地图组大概八九十人，点云融合组不到十个人，主要做Apollo robotaxi和百度ANP（L2辅助驾驶）

### 二面
1. 纯问项目，无八股，无手撕
2. FAST-LIO2和LIO-MAPPING这两种LIO的方法，有什么区别，对应的covariance有什么区别，对优化过程有什么影响
3. LIO过程中，如果利用滑窗的方法，既想让滑窗覆盖的范围大一些，又想不让IMU积分时间太长，应该怎么办（参考VINS-MONO的方法，回去补课）

### 三面
1. 聊天，聊项目

## 文远知行
### 一面
1. 面了一个半小时，没做自我介绍，上来就介绍阿里的项目
2. 面试官是做决策的，聊了四十多分钟的决策， 给了很多场景，问这些场景下车辆应该考虑哪些因素做决策
3. 手撕：给一组二维点以及一个四元数，输出旋转之后的二维点，用Eigen实现
4. 计算sqrt(x)

### 二面
1. 项目、实习（实习问了将近半小时，觉得达摩院的方案做的不行）
2. 手撕：反转每对括号间的子串（[Leetcode 1190](https://leetcode.cn/problems/reverse-substrings-between-each-pair-of-parentheses/)）