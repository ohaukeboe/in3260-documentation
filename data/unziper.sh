#!/bin/sh

dir="pcs-data-merged"

# shopt -s globstar
for z in $dir/*/*/*nethint.log; do
  echo "$z"
  filename="${z%\_pc*}"
  cat "$z" >> "$filename"_nethint.log
  rm "$z"

done
