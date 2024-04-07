# %%
import cv2
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from pexelsapi.pexels import Pexels
from dotenv import load_dotenv
import requests
import numpy as np


from towhee import ops, pipe
# %%
# pip for video embeddings


# pipe_video = (pipe.input('video_path').map('video_path', 'frames', ops.video_decode.ffmpeg(sample_type='uniform_temporal_subsample', args={'num_samples': 12})).map('frames', 'vec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='video', device='cuda')).output('vec'))


# %%
# set CUDA_VISIBLE_DEVICES to 1

# %%
vid_path = os.path.join(os.getcwd(), 'static', '1.mp4')


def get_video_chunks(path,
                     chunk_size_sec=3.26):
    vid = cv2.VideoCapture(path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    chunk_size = int(chunk_size_sec * fps)
    chunks = []
    chunk = []
    n_chunks = int(total_frames // chunk_size)
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Chunk size: {chunk_size}")

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        chunk.append(frame)
        if len(chunk) == chunk_size:
            chunks.append(np.array(chunk))
            chunk = []
    return chunks, fps, vid.get(cv2.CAP_PROP_FRAME_HEIGHT), vid.get(cv2.CAP_PROP_FRAME_WIDTH)




def write_chunks_as_video(chunk, output_path,
                          fps, 
                          height, width):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(width), int(height)))
    # for chunk in chunks:
    for frame in chunk:
        out.write(frame)
    out.release()
    
    
def chunks_paths(video_path, chunk_size_sec=3.25):
    
    chunks, fps, height, width = get_video_chunks(video_path, chunk_size_sec)
    chunk_paths = []
    vid_path_without_ext = video_path.split('.')[0]
    for i, chunk in enumerate(chunks):
        chunk_path = f'{vid_path_without_ext}_chunk_{i}.mp4'
        write_chunks_as_video(chunk, chunk_path, fps, height, width)
        chunk_paths.append(chunk_path)
        
    return chunk_paths

chunks_path = chunks_paths(vid_path)


def test_chunks_path_output_frames(chunks_path):
    for chunk_path in chunks_path:
        vid = cv2.VideoCapture(chunk_path)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in {chunk_path}: {total_frames}")
        vid.release()
        
test_chunks_path_output_frames(chunks_path)
# %%
pipe_video = (
    pipe.input('video_path')
    .map('video_path', 'frames_np_array', ops.video_decode.ffmpeg(sample_type='uniform_temporal_subsample', args={'num_samples': 12}))
    .map('frames_np_array', 'vec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='video', device='cuda'))
    .output('vec')
)
embeds = [pipe_video(chunk).get() for chunk in chunks_path]


# %%

pipe_text = (
    pipe.input('sentence')
    .map('sentence', 'vec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='text', device='cuda'))
    .output('vec')
)

sentence_embed = pipe_text('solar system').get()
sentence_embeds = [sentence_embed for _ in range(len(embeds))]



# %%

load_dotenv()
PEXELS_API_KEY = "gszGv4rBnHq1X62jwOc73uOhS0resHHzdHNPGlbU9DkjYSQtniki3bbp"
DOWNLOAD_DIR = './static'


pexel = Pexels(PEXELS_API_KEY)
search_videos = pexel.search_videos(
    query='ocean', orientation='', size='', color='', locale='', page=1, per_page=5)
print(search_videos)
# %%
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
