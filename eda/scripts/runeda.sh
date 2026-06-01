# Base direcotry
cd /home/darshan/MS

# Activate venv
conda init
conda activate venv

# RUN EDA FOR ISBI
python eda/scripts/exploratory/isbi_eda.py

# RUN EDA FOR MSLEGSEG
python eda/scripts/exploratory/msleg_eda.py

# RUN EDA FOR MICCAI
python eda/scripts/exploratory/miccai_eda.py



