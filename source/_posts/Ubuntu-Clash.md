---
title: Ubuntu系统中Clash的使用
date: 2023-02-16 12:51:47
tags: [Ubuntu, Clash, 教程]
categories: 工具使用
math: false
excerpt: 介绍如何利用Clash工具在Ubuntu系统中实现网络代理
---
# Clash安装
在Clash的[Github仓库](https://github.com/Dreamacro/clash/releases)中找到最新的Release，下载到电脑中并解压

{% asset_img clash.png figure %}

# 下载配置文件
在对应的机场中下载配置文件，笔者此处使用的是[桔子云](https://juzi20.com)，将下载的配置文件修改名称为{% label primary @config.yaml %}。下载[Country.mmdb](https://cdn.jsdelivr.net/gh/Dreamacro/maxmind-geoip@release/Country.mmdb)，将这两个文件放在~/.config/clash路径中

{% asset_img cofig.png figure %}

# 运行
```bash
cd ~/Softwares/clash
chmod 777 ./clash-linux-amd64-v3
./clash-linux-amd64-v3
```
在浏览器中打开[https://clash.razord.top](https://clash.razord.top)即可进入Clash的配置页面，可以实现修改节点，测试等功能

{% asset_img razord.png figure %}

# 全局配置
在Ubuntu的{% label primary @Setting->Network->Network Proxy %}中修改配置为手动，具体参数如下图。此时即完成了网络代理的设置

{% asset_img proxy.png figure %}