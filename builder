#! /bin/bash

command=$1


export CCC_CC=clang-10
export CCC_CXX=clang++-10
export CC=clang-10
export CXX=clang++-10

ROOT_DIR=`pwd`
BUILD_DIR=$ROOT_DIR/build

scanbuild=scan-build-10

function compile() {
    pushd $BUILD_DIR
    ninja
    popd
}

function build() {
    pushd $BUILD_DIR
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug $ROOT_DIR
    ninja -j `nproc`
    popd
}

function build-release() {
    pushd $BUILD_DIR
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release $ROOT_DIR
    ninja -j `nproc`
    popd
}

function clean() {
    rm -rf $BUILD_DIR
}

function run() {
    pushd $BUILD_DIR
    $BUILD_DIR/bin/serene "$@"
    popd
}

function memcheck() {
    pushd $BUILD_DIR
    ctest -T memcheck
    popd
}

function tests() {
    pushd $BUILD_DIR
    ctest
    popd
}


case "$command" in
    "deps")
        sudo apt update
        sudo apt install -y llvm-10 llvm-10-tools \
             clang-10 clang-format-10 clang-tidy-10 \
             clang-tools-10 valgrind cmake ninja-build
        ;;
    "build")
        clean
        mkdir -p $BUILD_DIR
        build
        ;;
    "build-release")
        clean
        mkdir -p $BUILD_DIR
        build
        ;;
    "compile")
        compilep
        ;;
    "run")
        run "$@"
        ;;

    "scan-build")
        clean
        mkdir -p $BUILD_DIR
        pushd $BUILD_DIR
        exec $scanbuild cmake $ROOT_DIR
        exec $scanbuild scan-build make -j 4
        popd
        ;;
    "memcheck")
        memcheck
        ;;
    "memcheck")
        tests
        ;;
    "clean")
        rm -rf $BUILD_DIR
        ;;
    "full-build")
        clean
        mkdir -p $BUILD_DIR
        build
        tests
        memcheck
        ;;
    *)
        echo "Commands: "
        echo "full-build - Build and test Serene."
        echo "build - Build Serene from scratch in DEBUG mode."
        echo "build-release - Build Serene from scratch in RELEASE mode."
        echo "compile - reCompiles the project using the already exist cmake configuration"
        echo "run - Runs the serene executable"
        echo "scan-build - Compiles serene with static analyzer"
        echo "tests - Runs the test cases"
        echo "memcheck - Runs the memcheck tool."
        echo "clean - :D"
        ;;
esac
