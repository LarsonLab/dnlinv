#!/bin/bash

for COUNT in {1..20}; do
	cd P$COUNT
	mkdir data
	extract single slice
	bart fft -u -i 1 kspace data/tmp_fft
	bart slice 0 160 data/tmp_fft data/full
	
	# Clean up complete raw data
	rm data/tmp_fft{.cfl,.hdr}
	rm GEheader.txt kspace{.cfl,.hdr} params.txt
	cd ..
done
