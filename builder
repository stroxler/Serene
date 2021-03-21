#! /bin/bash

command=$1


export CCC_CC=clang
export CCC_CXX=clang++
export CC=clang
export CXX=clang++

ROOT_DIR=`pwd`
BUILD_DIR=$ROOT_DIR/build

scanbuild=scan-build-11

function pushed_build() {
    pushd $BUILD_DIR > /dev/null
}

function popd_build() {
    popd > /dev/null
}

function compile() {
    pushed_build
    ninja
    popd_build
}

function build() {
    pushed_build
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug "$@" $ROOT_DIR
    ninja -j `nproc`
    popd_build
}

function build-release() {
    pushed_build
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release $ROOT_DIR
    ninja -j `nproc`
    popd_build
}

function clean() {
    rm -rf $BUILD_DIR
}

function run() {
    pushed_build
    $BUILD_DIR/bin/serene "$@"
    popd_build
}

function memcheck() {
    pushed_build
    ctest -T memcheck
    popd_build
}

function tests() {
    pushed_build
    ctest
    popd_build
}


case "$command" in
    # We need to fix this to have some instructions about how to build the dependencies
    "deps")
        echo "You need the following dependencies"
        echo "sudo apt update"
        echo "sudo apt install -y llvm-10 llvm-10-tools clang-10 clang-format-10 clang-tidy-10 clang-tools-10 valgrind cmake ninja-build doxygen"
        ;;
    "setup")
        pushd ./scripts
        ./scripts/git-pre-commit-format install
        popd
        ;;
    "build")
        clean
        mkdir -p $BUILD_DIR
        build "${@:2}"
        ;;
    "build-release")
        clean
        mkdir -p $BUILD_DIR
        build "${@:2}"
        ;;
    "compile")
        compile
        ;;
    "run")
        run "${@:2}"
        ;;

    "scan-build")
        clean
        mkdir -p $BUILD_DIR
        pushed_build
        exec $scanbuild cmake $ROOT_DIR
        exec $scanbuild scan-build make -j 4
        popd_build
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
