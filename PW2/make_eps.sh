#!bin/bash

for i in $(find report/pics -name "*.pdf")
do
	pdftops -eps ${i} "${i%.*}.eps"
done
