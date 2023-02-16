---
title: docker安装及使用
date: 2023-02-12 22:44:05
tags: [docker, ROS, Ubuntu]
categories: 工具使用
math: false
excerpt: 介绍在Ubuntu系统中安装Docker的方法，以及在Docker中实现ROS程序的开发和运行
---

# Docker安装
 - 使用apt方式安装docker：
```bash
sudo apt update
sudo apt install ca-certificates curl gnupg lsb-release
```

 - 添加阿里云的软件源
```bash
 curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```

 - 向sources.list添加Docker软件源
```bash
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```
 - 更新软件源，并安装Docker

```bash
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```
# Docker测试
## 启动Docker
```bash
sudo systemctl enable docker
sudo systemctl start docker
```
## 建立Docker用户组
将要使用Docker的用户添加到Docker的用户组，也就是默认的登录账户
```bash
# 建立Docker组
sudo groupadd docker
# 将当前用户添加到Docker组中
sudo usermod -aG docker $USER 
```
退出当前终端并重新登录（需要重启系统更新一下），进行如下测试

## 测试Docker是否正常安装
```bash
docker run --rm hello-world
```

如果出现一下信息，则说明安装成功
```bash
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
2db29710123e: Pull complete 
Digest: sha256:6e8b6f026e0b9c419ea0fd02d3905dd0952ad1feea67543f525c73a0a790fefb
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```

# Docker使用
## 获取Docker镜像

[Docker Hub](https://hub.docker.com/search?q=&type=image)上有大量的高质量镜像可以直接使用，此处以ROS的镜像为例。
```bash
# 拉取镜像
docker pull osrf/ros:melodic-desktop-full
```
## Docker启动
```bash
docker run -it osrf/ros:melodic-desktop-full
```
此时即启动了一个bash终端，允许用户进行交互

{% asset_img docker_bash.png figure %}

## 挂载主机目录
```bash
# 使用-v参数，如果本地目录不存在，则Docker会自动创建一个文件夹
docker run -it -v /home/jiashi/Src:/home/Src osrf/ros:melodic-desktop-full

# 使用--mount参数，如果本地目录不存在，Docker会报错（推荐）
docker run -it --mount type=bind,source=/home/jiashi/Src/,target=/home/Src osrf/ros:melodic-desktop-full
```

## Docker网络配置
通常情况下不同终端进入同一个ROS容器时，之间的信息是不互通的，相当于两个ROS Master，可以在运行Docker时添加{% label primary @--network host %}参数，实现Docker网络与本机互通。

```bash
docker run -it --network host osrf/ros:melodic-desktop-full
```

## ROS Docker图形化界面配置
[参考链接](http://wiki.ros.org/docker/Tutorials/GUI#The_simple_way)
```bash
docker run -it --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" osrf/ros:melodic-desktop-full
export containerId=$(docker ps -l -q)
```
此时通常会出现以下报错
```bash
No protocol specified
rqt: cannot connect to X server unix:0
```
再执行一下命令即可
```bash
xhost +local:root # for the lazy and reckless
```

# 常见问题
## Docker容器无法Tab补全
[参考文章](http://www.manongjc.com/detail/21-nsuquklxusjiofr.html)
Docker中新增了{% label primary @/etc/apt/apt.conf.d/docker-clean %}用于清除apt缓存来减小容器体积，因此需要删除该文件并重新安装{% label primary @bash-completion %}
```bash
rm etc/apt/apt.conf.d/docker-clean
sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
apt update
apt install -y bash-completion vim
vim /etc/bash.bashrc
```

将该文件中35-41行取消注释

```bash
# enable bash completion in interactive shells
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi
```
```bash
source /etc/bash.bashrc
```
此时即可正常Tab补全