#!/bin/bash
wget -O 200dim_dict.txt "https://www.dropbox.com/s/j622jw78jdbzgo0/200dim_dict.txt?dl=1"
wget -O hw4.h5 "https://www.dropbox.com/s/8y7chhtdlhtfqh8/hw4.h5?dl=1"
python main.py --test --model GRU --mode classify --prefix hw4 --maxlen 45 --test_file $1 --result_file $2
