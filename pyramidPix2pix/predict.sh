#!/bin/bash

python test.py --dataroot ./input/images/organelles-transmitted-light-ome-tiff --name light2Tubulin  
python test.py --dataroot ./input/images/organelles-transmitted-light-ome-tiff --name light2Nucleus  
python test.py --dataroot ./input/images/organelles-transmitted-light-ome-tiff --name light2Mitochondria  
python test.py --dataroot ./input/images/organelles-transmitted-light-ome-tiff --name light2Actin  

#python postprocess.py
