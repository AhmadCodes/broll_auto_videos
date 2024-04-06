cd /workspace
rm miniconda.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/miniconda.sh
bash minicondda.sh -b -p /root/miniconda3/ -f
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
cd /root/miniconda3/bin 
./conda init bash
source ~/.bashrc
cd /workspace/BRolls--whisper-sdxl
conda env create -f environment.yaml
conda activate app
apt update && apt install ffmpeg -y
apt install -qq imagemagick
pip install --quiet git+https://github.com/Zulko/moviepy.git@bc8d1a831d2d1f61abfdf1779e8df95d523947a5
pip install --quiet imageio==2.25.1
echo "python gradio_app.py"
python gradio_app.py
