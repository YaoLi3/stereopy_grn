#!/bin/bash
date >__start__
/home/softwares/PGM-1.0.0/PGM /dellfsqd2/ST_OCEAN/USER/liyao1/tools/anaconda3/envs/test/bin/python /dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/grn.py 
#/home/softwares/PGM-1.0.0/PGM /bin/bash test.sh
date > __end__

/home/softwares/PGM-1.0.0/report.sh > report.txt

rm pglog*

sed -i '/^\ touch/d' nohup.out
sed -i '/^\ start/d' nohup.out

