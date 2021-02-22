#! /usr/bin/bash env

cd ../../
PYTHONPATH=. python data_tools/build_file_list.py customdata data/customdata/rawframes/ --level 2 --format rawframes --shuffle
echo "Filelist for rawframes generated."

cd data_tools/customdata/