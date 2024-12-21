---
layout: post
title: Ubuntu使用Proxychains实现终端代理
date: 2024-02-17 20:04:48
tags: [Ubuntu, 工具, 科学上网, 代理]
categories: 工具使用
math: false
excerpt: Ubuntu使用Proxychains实现终端代理
---

# 安装Proxychains

> sudo apt install tor proxychains

# 配置Proxychains

- 修改/etc/proxychains.conf文件，添加socket5的代理信息，记得端口号和实际使用的保持一致：
<p align="center">{% asset_img config.png %}</p>

# 使用Proxychains
- 在需要走代理的命令前加上proxychains，比如：
> proxychains curl www.google.com