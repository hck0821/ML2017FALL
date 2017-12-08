#!/bin/bash
wget -O 200dim_dict.txt "https://www.dropbox.com/s/j622jw78jdbzgo0/200dim_dict.txt?dl=1"
wget -O hw4.h5 "https://www.dropbox.com/s/8y7chhtdlhtfqh8/hw4.h5?dl=1"
python main.py --train --model GRU --mode classify --prefix hw4 --train_file $1 --nolabel_file $2
