#!/bin/sh

for dir in *_common; do
    base=${dir%_common}
    mkdir -p "$base"
    mv "${base}_common" "${base}_nocommon" "${base}_wifi" "$base"
done
