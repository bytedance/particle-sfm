#!/bin/bash

# download MiDaS model
wget https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt
mv midas_v21-f6b98070.pt third_party/MiDaS/weights/

# download RAFT model
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip
rm models.zip
mv models third_party/RAFT/

