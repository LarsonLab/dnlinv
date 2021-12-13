#!/bin/bash

for COUNT in {1..20}; do
	wget http://old.mridata.org/knees/fully_sampled/p"$COUNT"/e1/s1/P"$COUNT".zip
done
