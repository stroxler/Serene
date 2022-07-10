FROM docker.io/debian:sid-slim

ARG VERSION

RUN apt-get update &&  \
    apt-get install --no-install-recommends -y \
    gnupg \
    cmake \
    ccache \
    git \
    ninja-build \
    binutils \
    lsb-release \
    wget \
    software-properties-common \
    zlib1g \
    cppcheck \
    shellcheck \
    zlib1g-dev

RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh
RUN ./llvm.sh ${VERSION} all
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    mlir-${VERSION}-tools \
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
ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++

# --branch clang_${VERSION}
WORKDIR /iwuy
RUN git clone https://github.com/include-what-you-use/include-what-you-use.git --depth 1 && \
    mkdir build && cd build && \
    cmake -G Ninja -DCMAKE_PREFIX_PATH=/usr/lib/llvm-${VERSION} ../include-what-you-use && \
    cmake --build . && \
    cmake -P cmake_install.cmake

WORKDIR /boehm
RUN git clone https://github.com/ivmai/bdwgc.git --depth 1 --branch v8.2.0 && \
    mkdir build && cd build && \
    cmake -G Ninja -DBUILD_SHARED_LIBS=OFF -Denable_cplusplus=ON -Denable_threads=ON \
    -Denable_gcj_support=OFF -Dinstall_headers=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON ../bdwgc && \
    cmake --build . --config Release && \
    cmake -P cmake_install.cmake

WORKDIR /app
RUN rm /iwuy -rf && rm /boehm -rf && rm /llvm.sh

RUN apt-get purge -y git software-properties-common wget && \
    apt-get autoremove -y && \
    apt-get clean
