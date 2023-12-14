datadir='final-data-over2'

# plot() {
#     ./plotter.py $(find $datadir -maxdepth 1 -mindepth 1 $finder) -Fvf owd -o "$outfile" > ${outfile%.pdf}.txt
# }

plot() {
    ./plotter.py $files -Fvf owd -o "$outfile" > ${outfile%.pdf}.txt
}

plot_legend() {
    ./plotter.py $files -l "$legend" -Fvf owd -o "$outfile" > ${outfile%.pdf}.txt
}

plot_corr() {
    ./plotter.py $files -CFvf owd -o "$outfile" > ${outfile%.pdf}.txt
}

# All

# outfile='figures/all-over2-no-only-video.pdf'
# files=$(ls -d $datadir/* | grep -v 'video$')
# plot

# outfile='figures/all-over2-no-video.pdf'
# files=$(ls -d $datadir/* | grep -v 'video')
# plot


# outfile='figures/double-cdf-all-over2-no-only-video.pdf'
# files=$(ls -d $datadir/* | grep -v 'video$')
# plot_corr

# outfile='figures/double-cdf-all-over2-no-video.pdf'
# files=$(ls -d $datadir/* | grep -v 'video')
# plot_corr

# # Only video

# video_dir='video-data'
# outfile='figures/only-video.pdf'
# files=$(ls -d $video_dir/*)
# legend='lower right'
# plot_legend


# ------------------------------------

# For each delay

# outfile='figures/double-cdf-delay-50-no-only-video.pdf'
# finder=' -name "*del25*" -and ( -name "*bbr*" -or -name "*reno*" -or -name "*cubic*" ) '
# ./plotter.py $(find $datadir -maxdepth 1 -mindepth 1 $finder) -Fvf owd -o "$outfile" > ${outfile%.pdf}.txt
# plot_corr

# outfile='figures/double-cdf-delay-10-no-only-video.pdf'
# finder=' -name "*del5*" -and ( -name "*bbr*" -or -name "*reno*" -or -name "*cubic*" ) '
# plot_corr

# # for each BDP combination

# outfile='figures/double-cdf-bdp-05-no-only-video.pdf'
# finder=' ( -name "*bs_6*" -or -name "*bs_31*" ) -and ( -name "*bbr*" -or -name "*reno*" -or -name "*cubic*" ) '
# plot_corr

# outfile='figures/double-cdf-bdp-10-no-only-video.pdf'
# finder=' ( -name "*bs_12*" -or -name "*bs_62*" ) -and ( -name "*bbr*" -or -name "*reno*" -or -name "*cubic*" ) '
# plot_corr

# outfile='figures/double-cdf-bdp-15-no-only-video.pdf'
# finder=' ( -name "*bs_18*" -or -name "*bs_93*" ) -and ( -name "*bbr*" -or -name "*reno*" -or -name "*cubic*" ) '
# plot_corr

# outfile='figures/double-cdf-bdp-20-no-only-video.pdf'
# finder=' ( -name "*bs_24*" -or -name "*bs_124*" ) -and ( -name "*bbr*" -or -name "*reno*" -or -name "*cubic*" ) '
# plot_corr

# outfile='figures/delay-50-no-only-video.pdf'
# finder=' -name "*del25*" -and ( -name "*bbr*" -or -name "*reno*" -or -name "*cubic*" ) '
# ./plotter.py $(find $datadir -maxdepth 1 -mindepth 1 $finder) -Fvf owd -o "$outfile" > ${outfile%.pdf}.txt
# plot

# outfile='figures/delay-10-no-only-video.pdf'
# finder=' -name "*del5*" -and ( -name "*bbr*" -or -name "*reno*" -or -name "*cubic*" ) '
# plot

# for each BDP combination


outfile='figures/bdp-05-no-video.pdf'
files=$(ls -d $datadir/* | grep -E 'bs_6_|bs_31_' | grep -v video)
legend='lower right'
plot_legend

outfile='figures/bdp-10-no-video.pdf'
# finder=' ( -name "*bs_12*" -or -name "*bs_62*" ) -and ( -not -name "*video*" ) '
files=$(ls -d $datadir/* | grep -E 'bs_12_|bs_62_' | grep -v video)
legend='lower right'
plot_legend

outfile='figures/bdp-15-no-video.pdf'
# finder=' ( -name "*bs_18*" -or -name "*bs_93*" ) -and ( -not -name "*video*" ) '
files=$(ls -d $datadir/* | grep -E 'bs_18_|bs_93_' | grep -v video)
legend='lower right'
plot_legend

outfile='figures/bdp-20-no-video.pdf'
# finder=' ( -name "*bs_24*" -or -name "*bs_124*" ) -and ( -not -name "*video*" ) '
files=$(ls -d $datadir/* | grep -E 'bs_24_|bs_124_' | grep -v video)
legend='lower right'
plot_legend



# ------------------------------------

# All No only video

# # filename='*'
# outfile='figures/all-over2-noonly-video.pdf'
# title='All (with over 2 data points, no only video)'
# ./plotter.py $(find $datadir -maxdepth 1 -mindepth 1 -name "*bbr*" -or -name "*reno*" -or -name "*cubic*" ) -t "$title" -Fvf owd -o "$outfile" > ${outfile%.pdf}.txt


# Different no video plots

# filename='*3bbr*'
# outfile='figures/3bbr_novideo.pdf'
# title='3BBR (no video)'
# ./plotter.py $(find $datadir -maxdepth 1 -mindepth 1 -name "$filename" -and -not -name '*video*') -t "$title" -Fvf owd -o "$outfile" > ${outfile%.pdf}.txt

# filename='*3reno*'
# outfile='figures/3reno_novideo.pdf'
# title='3RENO (no video)'
# ./plotter.py $(find $datadir -maxdepth 1 -mindepth 1 -name "$filename" -and -not -name '*video*') -t "$title" -Fvf owd -o "$outfile" > ${outfile%.pdf}.txt

# filename='*3cubic*'
# outfile='figures/3cubic_novideo.pdf'
# title='3CUBIC (no video)'
# ./plotter.py $(find $datadir -maxdepth 1 -mindepth 1 -name "$filename" -and -not -name '*video*') -t "$title" -Fvf owd -o "$outfile" > ${outfile%.pdf}.txt

# filename='*1bbr_1cubic_1reno*'
# outfile='figures/1bbr_1cubic_1reno_novideo.pdf'
# title='BBR CUBIC and RENO (no video)'
# ./plotter.py $(find $datadir -maxdepth 1 -mindepth 1 -name "$filename" -and -not -name '*video*') -t "$title" -Fvf owd -o "$outfile" > ${outfile%.pdf}.txt

# Different with video plots

# filename='*3bbr*'
# outfile='figures/3bbr_o2.pdf'
# title='3BBR (over 2 data points)'
# ./plotter.py $(find $datadir -maxdepth 1 -mindepth 1 -name "$filename") -t "$title" -Fvf owd -o "$outfile" > ${outfile%.pdf}.txt

# filename='*3reno*'
# outfile='figures/3reno_o2.pdf'
# title='3RENO (over 2 data points)'
# ./plotter.py $(find $datadir -maxdepth 1 -mindepth 1 -name "$filename") -t "$title" -Fvf owd -o "$outfile" > ${outfile%.pdf}.txt

# filename='*3cubic*'
# outfile='figures/3cubic_o2.pdf'
# title='3CUBIC (over 2 data points)'
# ./plotter.py $(find $datadir -maxdepth 1 -mindepth 1 -name "$filename") -t "$title" -Fvf owd -o "$outfile" > ${outfile%.pdf}.txt

# filename='*1bbr_1cubic_1reno*'
# outfile='figures/1bbr_1cubic_1reno_o2.pdf'
# title='BBR CUBIC and RENO (over 2 data points)'
# ./plotter.py $(find $datadir -maxdepth 1 -mindepth 1 -name "$filename") -t "$title" -Fvf owd -o "$outfile" > ${outfile%.pdf}.txt



# datadir='final-data'

# # ALL
# filename='*'
# outfile='figures/all.pdf'
# title='All data'
# ./plotter.py $(find $datadir -maxdepth 1 -mindepth 1 -type d) -t "$title" -Fvf owd -o "$outfile" > ${outfile%.pdf}.txt
