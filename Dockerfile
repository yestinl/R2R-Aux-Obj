# Matterport3DSimulator
# Requires nvidia gpu with driver 384.xx or higher


FROM nvidia/cudagl:9.0-devel-ubuntu16.04

# Install a few libraries to support both EGL and OSMESA options
RUN apt-get install -y sed
#RUN sed -r 's;((archive)|(security)).ubuntu.com;mirrors.aliyun.com;g' /etc/apt/sources.list -i
RUN sed -r 's;((archive)|(security)).ubuntu.com;mirrors.tuna.tsinghua.edu.cn/ubuntu;g' /etc/apt/sources.list -i
RUN apt-get update -y && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
# RUN add-apt-repository -r ppa:jonathonf/python-3.6
RUN apt-get update -y
RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv && \
        apt-get install -y git
RUN apt-get update && apt-get install -y wget doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python-opencv python-setuptools
RUN apt-get install -y tmux vim libopencv-dev freeglut3 freeglut3-dev

# update pip
RUN python3.6 -m pip install pip --upgrade && \
        python3.6 -m pip install wheel

# RUN echo "nameserver 192.168.3.161" >> /etc/resolv.conf && \
#     echo "nameserver 8.8.8.8" >> /etc/resolv.conf && \
#     echo "nameserver 8.8.4.4" >> /etc/resolv.conf
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# RUN pip install 'tensorflow-gpu'

#install latest cmake
# ADD /build/cmake-3.12.2-Linux-x86_64.sh /cmake-3.12.2-Linux-x86_64.sh
ADD cmake-3.17.0-Linux-x86_64.sh /cmake-3.17.0-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.17.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

ENV HOME="/home"

RUN ["apt-get", "install", "-y", "git"]

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install cffi cython lockfile

RUN pip uninstall -y 'numpy'
RUN pip install numpy==1.16.4

RUN apt-get install libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev

RUN alias python=python3.6
RUN echo 'alias python=python3.6' >> ~/.bashrc
RUN pip install tensorboardX tqdm
RUN python3.6 -m pip install polyaxon-client==0.5.3
RUN pip3 install torch==1.4.0 torchvision==0.5.0 -i https://pypi.doubanio.com/simple/
RUN apt-get -y install sshfs

# RUN make install
# use zsh when docker run
# CMD [ "zsh" ]
RUN pip3 install tensorboard
RUN pip3 install boto3
RUN pip3 install requests

ARG CACHEBUST=1
#RUN git clone https://github.com/airsplay/R2R-EnvDrop.git
#ADD /build/ /R2R-EnvDrop
ADD R2R-Aux /R2R-Aux
WORKDIR R2R-Aux
# RUN git checkout final
ADD python_requirements.txt /python_requirements.txt
RUN pip install -r /python_requirements.txt
RUN git submodule update --init --recursive

RUN apt-get install libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev
RUN mkdir build
WORKDIR build
RUN cmake -DEGL_RENDERING=ON ..
RUN make -j8

RUN echo 'export MATTERPORT_DATA_DIR='`pwd` >> ~/.bashrc
# RUN source ~/.bashrc
RUN apt-get install rsync -y

# WORKDIR /code
# COPY build /code
