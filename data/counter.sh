#!/bin/sh

# dir="pcs-data-merged"
# dir="router-data"
dir="pcs-data"

./plotter.py -Ccn -t "web" $dir/*web* &
./plotter.py -Ccn -t "not web" $(find $dir -mindepth 1 -maxdepth 1 -type d -not -name "*web*") &
./plotter.py -Ccn -t "not web nor video" $(find $dir -mindepth 1 -maxdepth 1 -type d -not -name "*web*" -and -not -name "*video*") &
./plotter.py -Ccn -t "not web and with bbr, cubic, and/or reno" $(find $dir -mindepth 1 -maxdepth 1 -type d -not -name "*web*" -and -name "*bbr*" -or -name "*cubic*" -or -name "*reno*") &
