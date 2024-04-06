# %%
import os
import numpy as np



from towhee import ops, pipe
# pip for video embeddings
pipe_video = (
    pipe.input('video_path')
    .map('video_path', 'frames', ops.video_decode.ffmpeg(sample_type='uniform_temporal_subsample', args={'num_samples': 12}))
    .map('frames', 'vec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='video', device='cuda'))
    # .map('vec', 'vec', ops.normalize)
    .output('vec')
)
# %%
pipe_video("/workspace/broll_videos/app/static/test.mp4").get()

# %%

pipe_text = (
    pipe.input('sentence')
    .map('sentence', 'vec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='text', device='cuda'))
    .output('vec')
)
# %%

import os
import requests
from pexels_api import API
from dotenv import load_dotenv
load_dotenv()
PEXELS_API_KEY = "gszGv4rBnHq1X62jwOc73uOhS0resHHzdHNPGlbU9DkjYSQtniki3bbp"
DOWNLOAD_DIR = './static'


from pexelsapi.pexels import Pexels
pexel = Pexels(PEXELS_API_KEY)
search_videos = pexel.search_videos(query='ocean', orientation='', size='', color='', locale='', page=1, per_page=5)
print(search_videos)
#%%
for video in search_videos["videos"]:
    video_id = video["id"]
    download_url = f'https://www.pexels.com/video/{video_id}/download'
    response = requests.get(download_url)
    
    if response.status_code == 200:
        file_name = os.path.join(DOWNLOAD_DIR, f'{video_id}.mp4')
        with open(file_name, 'wb') as file:
            file.write(response.content)
        print(f'Downloaded video: {file_name}')
    else:
        print(f'Failed to download video: {video_id}')
# %%
