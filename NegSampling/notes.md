# MIMIC file length distribution

for file in *.txt; do cat $file | uniq | wc -w ; done | sort -g > ~/Temp/counts.txt
