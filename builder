#! /bin/bash

command=$1

# Utilize `ccache` if available
if type "ccache" > /dev/null
then
    CC="$(which ccache) $(which clang)"
    CXX="$(which ccache) $(which clang++)"
else
    CC=$(which clang)
    CXX=$(which clang++)
fi

export CC
export CXX

# Meke sure to use `lld` linker it faster and has a better UX
export LDFLAGS="-fuse-ld=lld"
export ASAN_FLAG="-fsanitize=address"
export ASAN_OPTIONS=check_initialization_order=1
LSAN_OPTIONS=suppressions=$(pwd)/.ignore_sanitize
export LSAN_OPTIONS

# The `builder` script is supposed to be run from the
# root of the source tree
ROOT_DIR=$(pwd)
BUILD_DIR=$ROOT_DIR/build

scanbuild=scan-build

function pushed_build() {
    pushd "$BUILD_DIR" > /dev/null || return
}

function popd_build() {
    popd > /dev/null || return
}

function compile() {
    pushed_build
    ninja -j "$(nproc)"
    popd_build
}

function build() {
    pushed_build
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug "$@" "$ROOT_DIR"
    ninja -j "$(nproc)"
    popd_build
}

function build-20() {
    pushed_build
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCPP_20_SUPPORT=ON "$@" "$ROOT_DIR"
    ninja -j "$(nproc)"
    popd_build
}

function build-release() {
    pushed_build
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release "$ROOT_DIR"
    ninja -j "$(nproc)"
    popd_build
}

function build-docs() {
    pushed_build
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Docs "$ROOT_DIR"
    ninja -j "$(nproc)"
    popd_build
}

function clean() {
    rm -rf "$BUILD_DIR"
}

function run() {
    pushed_build
    "$BUILD_DIR"/bin/serenec "$@"
    popd_build
}

function memcheck() {
    export ASAN_FLAG=""
    build
    pushed_build
    valgrind --tool=memcheck --leak-check=yes --trace-children=yes "$BUILD_DIR"/bin/serenec "$@"
    popd_build
}

function run-tests() {
    "$BUILD_DIR"/src/tests/tests
}

function tests() {
    pushed_build
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON "$ROOT_DIR"
    ninja -j "$(nproc)"
    popd_build
}


case "$command" in
    "setup")
        pushd ./scripts || return
        ./git-pre-commit-format install
        popd || return
	echo "=== Manual action required ==="
	echo "Set this environment variable. (clang-format-diff.py is located inside llvm's source directory)."
	echo 'export CLANG_FORMAT_DIFF="python3 /path/to/llvm/saurce/llvm-project/clang/tools/clang-format/clang-format-diff.py"'
        ;;
    "build")
        clean
        mkdir -p "$BUILD_DIR"
        build "${@:2}"
        ;;
    "build-20")
        clean
        mkdir -p "$BUILD_DIR"
        build-20 "${@:2}"
        ;;

    "build-docs")
        clean
        mkdir -p "$BUILD_DIR"
        build-docs "${@:2}"
        ;;

    "build-release")
        clean
        mkdir -p "$BUILD_DIR"
        build-release "${@:2}"
        ;;
    "compile")
        compile
        ;;
    "compile-and-test")
        compile
        run-tests
        ;;
    "run")
        run "${@:2}"
        ;;
    "run-tests")
        run-tests "${@:2}"
        ;;

    "scan-build")
        clean
        mkdir -p "$BUILD_DIR"
        pushed_build
        exec $scanbuild cmake "$ROOT_DIR"
        exec $scanbuild scan-build make -j 4
        popd_build
        ;;
    "memcheck")
        memcheck "${@:2}"
        ;;
    "tests")
        clean
        mkdir -p "$BUILD_DIR"
        tests
        run-tests
        ;;
    "clean")
        rm -rf "$BUILD_DIR"
        ;;
    "full-build")
        clean
        mkdir -p "$BUILD_DIR"
        build
        tests
        run-tests
        memcheck
        ;;
    *)
        echo "Commands: "
        echo "setup - Setup the githooks for devs"
        echo "full-build - Build and test Serene."
        echo "build - Build Serene from scratch in DEBUG mode."
        echo "build-release - Build Serene from scratch in RELEASE mode."
        echo "compile - reCompiles the project using the already exist cmake configuration"
        echo "compile-and-tests - reCompiles the project using the already exist cmake configuration and runs the tests"
        echo "run - Runs the serene executable"
        echo "scan-build - Compiles serene with static analyzer"
        echo "tests - Runs the test cases"
        echo "memcheck - Runs the memcheck(valgrind) tool."
        echo "clean - :D"
        ;;
esac
