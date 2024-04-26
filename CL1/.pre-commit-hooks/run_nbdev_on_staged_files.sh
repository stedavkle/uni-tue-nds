#!/usr/bin/env bash

for arg in "$@"; do
    nbdev_clean --fname "$arg"
done