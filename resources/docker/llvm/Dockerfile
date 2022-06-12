FROM debian:sid-slim

ARG VERSION

RUN uname -a
RUN apt-get update && apt-get install -y  apt-utils && apt-get install -y wget \
    gnupg \
    ccache \
    cmake \
    ccache \
    git \
    ninja-build \
    build-essential \
    binutils \
    lsb-release \
    wget \
    software-properties-common \
    zlib1g \
    apt-utils \
    cppcheck \
    shellcheck \
    zlib1g-dev

RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh
RUN ./llvm.sh ${VERSION} all
RUN apt-get update --fix-missing && \
    apt-get install -y mlir-${VERSION}-tools \
    libmlir-${VERSION}-dev \
    libmlir-${VERSION} \
    libmlir-${VERSION}-dbgsym \
    liblld-${VERSION} \
    liblld-${VERSION}-dev \
    clang-format-${VERSION} \
    clang-tidy-${VERSION}

RUN ln -s `which lld-${VERSION}` /usr/bin/lld && \
    ln -s `which clang-${VERSION}` /usr/bin/clang && \
    ln -s `which clang++-${VERSION}` /usr/bin/clang++ && \
    ln -s `which clang-format-${VERSION}` /usr/bin/clang-format && \
    ln -s `which clang-tidy-${VERSION}` /usr/bin/clang-tidy && \
    ln -s `which mlir-tblgen-${VERSION}`  /usr/bin/mlir-tblgen

ENV MLIR_DIR /usr/lib/llvm-${VERSION}
ENV CMAKE_PREFIX_PATH=/usr/lib/llvm-${VERSION}
ENV LD_LIBRARY_PATH=/usr/lib/llvm-${VERSION}/lib/clang/${VERSION}.0.0/lib/linux/

RUN apt-get clean

WORKDIR /iwuy
RUN git clone https://github.com/include-what-you-use/include-what-you-use.git --depth 1 && \
    mkdir build && cd build && \
    cmake -G "Unix Makefiles" -DCMAKE_PREFIX_PATH=/usr/lib/llvm-${VERSION} ../include-what-you-use && \
    make && make install

WORKDIR /app
RUN rm /iwuy -rf

COPY .pre-commit-config.yaml .
RUN git init .
RUN apt-get install -y --no-install-recommends pre-commit && pre-commit autoupdate && rm /app/.pre-commit-config.yaml
