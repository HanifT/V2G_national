#!/bin/bash

mkdir -p Data/NHTS_2017
mkdir -p Data/NHTS_2022
mkdir -p Data/Generated_Data

if [ ! -f Data/NHTS_2017/csv.zip ]; then
	curl -o Data/NHTS_2017/csv.zip https://nhts.ornl.gov/assets/2016/download/csv.zip
	echo "NHTS Data Downloaded"
else
	echo "NHTS Data Downloaded"
fi

if [ ! -f Data/NHTS_2017/trippub.csv ]; then
	unzip Data/NHTS_2017/csv.zip -d Data/NHTS_2017
	echo "NHTS Data Unzipped"
else
	echo "NHTS Data Unzipped"
fi

if [ ! -f Data/NHTS_2022/csv.zip ]; then
	curl -o Data/NHTS_2022/csv.zip https://nhts.ornl.gov/assets/2022/download/csv.zip
	echo "NHTS Data Downloaded"
else
	echo "NHTS Data Downloaded"
fi

if [ ! -f Data/NHTS_2022/trippub.csv ]; then
	unzip Data/NHTS_2022/csv.zip -d Data/NHTS_2022
	echo "NHTS Data Unzipped"
else
	echo "NHTS Data Unzipped"
fi