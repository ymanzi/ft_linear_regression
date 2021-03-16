# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ymanzi <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/10/09 14:05:13 by ymanzi            #+#    #+#              #
#    Updated: 2020/11/22 13:49:15 by ymanzi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# git clone https://github.com/Homebrew/brew ~/.linuxbrew/Homebrew
# mkdir ~/.linuxbrew/bin
# ln -s ~/.linuxbrew/Homebrew/bin/brew ~/.linuxbrew/bin
# eval $(~/.linuxbrew/bin/brew shellenv)

env:
	brew install python@3.8
	python -m pip install -U pip
	pip install numpy
	pip install pandas
	pip install matplotlib

clean:
	rm -f __pycache__
	python3.8 resources/init.py

predict:
	python3.8 predict.py

show:
	python3.8 show.py

train:
	python3.8 train.py