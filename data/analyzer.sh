#!/bin/sh

directory='final-data-under2'

# shopt -s globstar
for dir in $directory/*; do
    echo "$dir"

    for test in $dir/*; do
        # echo "Test: $test"

        for z in $test/*.dmp; do
            filename=${z%.dmp}_nethint.log
            echo $filename
            tmux split-window "python3 ~/projects/nethint/NETHINT/src/main.py --local -r $z -l $filename"

            # tmux split-window "echo $filename && sleep 10"
            # sed -i 's/}/}\n/g' $filename
            # tmux split-window "echo $filename && sleep 10" \; select-layout even-vertical
        done

    done

    tmux select-pane -t 0
    tmux select-layout even-vertical
    sleep 600
    tmux kill-pane -a

    for z in $dir/*/*.dmp; do
        filename=${z%.dmp}_nethint.log
        sed -i 's/}/}\n/g' $filename
        filename2="${filename%\_pc*}"
        cat "$filename" >> "$filename2"_nethint.log
        rm "$filename"
    done

    # read -p "Press Enter to continue"
done
