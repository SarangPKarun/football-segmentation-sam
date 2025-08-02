# football-segmentation-sam


use anaconda

conda env create -f environment.yml


install 
!git clone https://github.com/facebookresearch/sam2.git

%cd sam2

%pip install -e .

cd checkpoints && \
./download_ckpts.sh && \
cd ..



export FLASK_APP=app.py
export FLASK_ENV=development
flask run

watch -n 1 nvidia-smi    ---   to check the gpu usage if we run flask run again and again




video input in static upload


segment output in static
