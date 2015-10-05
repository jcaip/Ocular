#!/bin/bash
STRING="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-.:;<=>?[]^_"
echo "The Alphabet is: "
echo $STRING

n=0
maxjobs=4

for((i=0; i<${#STRING}; i++)); do
	curChar=${STRING:$i:1}
	echo $curChar
	while IFS='' read -r line || [[ -n "$line" ]]; do
		name="data_"$curChar"_"$line".gif"
    		`convert -gravity center -background white -fill black -font $line -size 20x20 label:$curChar $name`
	done < "$1" 	&
	if (( $(($((++n)) % $maxjobs)) == 0 )) ; then
        	wait # wait until all have finished (not optimal, but most times good enough)
        	echo "Batch completed"
    	fi
done

`python processImages.py`
find . -name '*.gif' -delete
