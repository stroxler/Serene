FROM debian:sid-slim

ARG VERSION

RUN apt-get update && apt-get install -y wget \
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
    zlib1g-dev

RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh
RUN ./llvm.sh ${VERSION} all
RUN apt-get update --fix-missing && \
    apt-get install -y mlir-${VERSION}-tools \
    libmlir-${VERSION}-dev \
    libmlir-${VERSION} \
    libmlir-${VERSION}-dbgsym \
    liblld-${VERSION} \
    liblld-${VERSION}-dev

RUN ln -s `which lld-${VERSION}` /usr/bin/lld && \
    ln -s `which clang-${VERSION}` /usr/bin/clang && \
    ln -s `which clang++-${VERSION}` /usr/bin/clang++ && \
    ln -s `which mlir-tblgen-${VERSION}`  /usr/bin/mlir-tblgen

ENV MLIR_DIR /usr/lib/llvm-${VERSION}
ENV CMAKE_PREFIX_PATH=/usr/lib/llvm-${VERSION}
ENV LD_LIBRARY_PATH=/usr/lib/llvm-${VERSION}/lib/clang/${VERSION}.0.0/lib/linux/

RUN apt-get clean
