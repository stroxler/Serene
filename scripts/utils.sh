#! /bin/bash
# Serene Programming Language
#
# Copyright (c) 2019-2022 Sameer Rahmani <lxsameer@gnu.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

set -e

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
function fn-names() {
    grep -E '^function [0-9a-zA-Z_-]+\(\) \{ ## .*$$' "$0" | sed 's/^function \([a-zA-Z0-9_-]*\)() { ## \(.*\)/\1/'
}

function info() {
    if [ "$1" ]
    then
        echo -e "[\033[01;32mINFO\033[00m]: $*"
    fi
}

function error() {
    if [ "$1" ]
    then
        echo -e "[\033[01;31mERR\033[00m]: $*"
    fi
}

function warn() {
    if [ "$1" ]
    then
        echo -e "[\033[01;33mWARN\033[00m]: $*"
    fi
}

function yes_or_no {
    while true; do
        read -rp "$* [y/n]: " yn
        case $yn in
            [Yy]*) return 0  ;;
            [Nn]*) echo "Aborted" ; return  1 ;;
        esac
    done
}
