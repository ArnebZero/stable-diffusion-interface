git clone https://github.com/CompVis/stable-diffusion
mkdir stable-diffusion/models/ldm/stable-diffusion-v1/
mv model.ckpt stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt
mv model_service.py config.yaml stable-diffusion/
pip install -r requirements.txt
cd stable-diffusion/
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
mkdir logs/
