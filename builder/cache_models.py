# builder/model_fetcher.py
# %%
from towhee import pipe, ops

import os
curr_file_dir = os.path.dirname(os.path.realpath(__file__))
cache_dir = os.path.join(curr_file_dir, "..", "cache")

MODEL_CACHE = cache_dir


def fetch_pretrained_model():
    video_embeddings_pipe = (
        pipe.input('video_path')
        .map('video_path', 'frames', ops.video_decode.ffmpeg(sample_type='uniform_temporal_subsample', args={'num_samples': 12}))
        .map('frames', 'vec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='video', device='cpu'))
        # .map('vec', 'vec', ops.normalize)
        .output('vec')
    )

    text_embeddings_pipe = (
        pipe.input('sentence')
        .map('sentence', 'vec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='text', device='cpu'))
        .output('vec')
    )


if __name__ == "__main__":
    fetch_pretrained_model()
# %%
