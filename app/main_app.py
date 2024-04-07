# %%

# from moviepy.config import change_settings
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

# change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

import cv2
import traceback as tb
from pexelsapi.pexels import Pexels
import uuid
from io import BytesIO
import boto3
from dotenv import load_dotenv
import requests
import json
from towhee import ops, pipe
import scipy as sc
import numpy as np
import os

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), "..", '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


def get_s3_client():
    s3_client = boto3.client('s3',
                             aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    return s3_client


# Set your Pexels API key
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# Create an API object
pexel_api = Pexels(PEXELS_API_KEY)

# Search for videos with the keyword 'ocean'


# %%

video_embeddings_pipe = (
    pipe.input('video_path')
    .map('video_path', 'frames', ops.video_decode.ffmpeg(sample_type='uniform_temporal_subsample', args={'num_samples': 12}))
    .map('frames', 'vec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='video', device='cuda'))
    # .map('vec', 'vec', ops.normalize)
    .output('vec')
)

text_embeddings_pipe = (
    pipe.input('sentence')
    .map('sentence', 'vec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='text', device='cuda'))
    .output('vec')
)

S3_CLIENT = get_s3_client()
# %%


def chunk_level_transcript(word_level_transcript, chunk_size_s=3):
    chunked_ts = []
    start_t = 0
    started = False
    end_t = 0
    chunk = []

    for i in range(0, len(word_level_transcript)):
        chunk.append(word_level_transcript[i])
        if not started:
            start_t = float(word_level_transcript[i]['start'])
            started = True
        end_t = float(word_level_transcript[i]['end'])
        delta = end_t - start_t
        if delta >= chunk_size_s:
            chunk_words_str = " ".join([w['word'] for w in chunk])
            chunk_start = chunk[0]['start']
            chunk_end = chunk[-1]['end']
            chunked_ts.append(
                {"start": chunk_start, "end": chunk_end, "segment": chunk_words_str})
            started = False
            chunk = []

    if len(chunk) > 0:
        chunk_words_str = " ".join([w['word'] for w in chunk])
        chunk_start = chunk[0]['start']
        chunk_end = chunk[-1]['end']
        chunked_ts.append(
            {"start": chunk_start, "end": chunk_end, "segment": chunk_words_str})
    return chunked_ts


def convert_to_srt(word_level_transcript):
    srt = ""
    for i, word in enumerate(word_level_transcript):
        srt += f"{i + 1}\n"
        srt += f"{word['start']} --> {word['end']}\n"
        srt += f"{word['segment']}\n\n"
    return srt


def convert_to_text(word_level_transcript):
    text = " ".join([w['word'] for w in word_level_transcript])
    return text


def give_context(word_level_transcript,
                 context_start_s,
                 context_end_s,
                 context_buffer_s=5):
    context = ""
    for word in word_level_transcript:
        if float(word['start']) >= context_start_s - context_buffer_s and float(word['end']) <= context_end_s + context_buffer_s:
            context += word['word'] + " "
    return context


# %%
chatgpt_url = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# %%


def get_video_chunks(path,
                     chunk_size_sec):
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


def get_chunks_paths(video_path, chunk_size_sec):

    chunks, fps, height, width = get_video_chunks(video_path, chunk_size_sec)
    chunk_paths_dicts = []
    vid_path_without_ext = video_path.split('.')[0]
    for i, chunk in enumerate(chunks):
        start_s = i * chunk_size_sec
        end_s = (i + 1) * chunk_size_sec
        chunk_path = f'{vid_path_without_ext}_chunk_{i}.mp4'
        write_chunks_as_video(chunk, chunk_path, fps, height, width)
        chunk_paths_dicts.append({"path": chunk_path,
                                  "start_s": start_s,
                                  "end_s": end_s})

    return chunk_paths_dicts


def download_pexels_video(search_phrases,
                          a_roll_start_s,
                          a_roll_end_s,
                          download_dir=os.path.join(
                              os.path.dirname(__file__), "static"),
                          n_searches_per_phrase=3,
                          debug=False):
    search_videos = []
    err_msg = ""
    downloaded_files = None
    try:
        for search_phrase in search_phrases:
            search_results = pexel_api.search_videos(
                query=search_phrase, orientation='', size='', color='', locale='', page=1, per_page=n_searches_per_phrase)
            search_videos.extend(search_results['videos'])
        if debug:
            print(search_videos)

        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        downloaded_files = {}

        for i, video in enumerate(search_videos):
            video_id = video["id"]
            v = video["video_files"]
            sd = [v_ for v_ in v if v_["quality"] == "sd"]
            if len(sd) == 0:
                hd = [v_ for v_ in v if v_["quality"] == "hd"]
                if len(hd) == 0:
                    continue
                else:
                    video_url = hd[0]["link"]
            else:
                video_url = sd[0]["link"]
            download_url = video_url
            response = requests.get(download_url)

            if response.status_code == 200:
                file_name = os.path.join(download_dir, f'{i}.mp4')
                with open(file_name, 'wb') as file:
                    file.write(response.content)
                chunk_size_sec = abs(float(a_roll_end_s) - a_roll_start_s)
                chunk_path_dicts = get_chunks_paths(file_name,
                                                    chunk_size_sec=chunk_size_sec)
                for chunk_path_dict in (chunk_path_dicts):
                    start_s = chunk_path_dict['start_s']
                    end_s = chunk_path_dict['end_s']
                    chunk_size_s = abs(end_s - start_s)
                    chunk_path = chunk_path_dict['path']
                    video_id = str(video_id) + "_" + \
                        str(start_s) + "_" + str(end_s)

                    downloaded_files[video_id] = {"local_path": chunk_path,
                                                  "b_roll_video_start_s": start_s,
                                                  "b_roll_video_end_s": end_s,
                                                  "a_roll_video_insertion_start_s": a_roll_start_s,
                                                  "a_roll_video_insertion_end_s": a_roll_end_s,
                                                  "b_roll_video_information": video,
                                                  }
                if debug:
                    print(f'Downloaded video: {file_name}')
            else:
                print(f'Failed to download video: {video_id}')
    except Exception as e:
        err_msg = "Error in downloading videos from Pexels: " + str(e)
        return downloaded_files, err_msg

    return downloaded_files, err_msg


def get_video_embeddings(video_path):
    return video_embeddings_pipe(video_path).get()


def get_text_embeddings(sentence):
    return text_embeddings_pipe(sentence).get()


def rank_videos(video_dict,
                sentence,
                top_K=5):
    sentence_embedding = get_text_embeddings(sentence)
    video_ids = list(video_dict.keys())
    video_paths = [video_dict[video_id]["local_path"]
                   for video_id in video_ids]

    ranked_videos = []

    video_embeddings = []
    for video_path in video_paths:
        video_embeddings.append(get_video_embeddings(video_path))

    video_embeddings = np.array(video_embeddings)
    video_embeddings = np.squeeze(video_embeddings)
    sentence_embedding = np.array([sentence_embedding])
    sentence_embedding = np.squeeze(sentence_embedding)

    # Compute the similarity between the sentence and each video using L2 distance
    similarity = sc.linalg.norm(video_embeddings - sentence_embedding, axis=1)

    ranked_video_paths = [[video_paths[i], similarity[i]]
                          for i in similarity.argsort()]

    for i, [r, s] in enumerate(ranked_video_paths):

        k = [k for k, v in video_dict.items() if v["local_path"] == r][0]
        video_dict[k]["Rank"] = i+1
        video_dict[k]["Distance"] = s
        video_dict[k]["B-roll description"] = sentence
        ranked_videos.append(video_dict[k])

    return ranked_videos[:top_K]


# %%

def validate_KV_pair(dict_list,
                     debug=False):
    for d in dict_list:
        check_all_keys = all([k in d.keys()
                             for k in ['description', "search_phrase", "start", "end"]])

        check_description = isinstance(d['description'], str)
        check_keywords = isinstance(d['search_phrase'], str)
        check_start = isinstance(d['start'], float)
        check_end = isinstance(d['end'], float)

        if debug:
            print("check_all_keys: ", check_all_keys)
            print("check_description: ", check_description)
            print("check_keywords: ", check_keywords)
            print("check_start: ", check_start)
            print("check_end: ", check_end)

        return check_all_keys and check_description and check_keywords and check_start and check_end


def json_corrector(json_str,
                   exception_str,
                   openaiapi_key):

    headers = {
        "content-type": "application/json",
        "Authorization": "Bearer {}".format(openaiapi_key)}

    prompt_prefix = f"""Exception: {exception_str}
    JSON:{json_str}
    ------------------
    """
    prompt = prompt_prefix + """\n Correct the following JSON, eliminate any formatting issues occured due to misplaces or lack or commas, brackets, semicolons, colons, other symbols, etc
    \nJSON:"""

    # Define the payload for the chat model
    messages = [
        {"role": "system", "content": "You are an expert in correcting JSON strings, you return a VALID JSON by eliminating all formatting issues"},
        {"role": "user", "content": prompt}
    ]
    chatgpt_payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "temperature": 1.3,
        "max_tokens": 2000,
        "top_p": 1,
        "stop": ["###"]
    }

    try:
        url = chatgpt_url
        response = requests.post(url, json=chatgpt_payload, headers=headers)
        response_json = response.json()

        try:
            print("response ",
                  response_json['choices'][0]['message']['content'])
        except:
            print("response ", response_json)

            return None
        # Extract data from the API's response
        try:
            output = json.loads(
                response_json['choices'][0]['message']['content'].strip())
            return output
        except Exception as e:
            print("Error in response from OPENAI GPT-3.5: ", e)
            return None

    except Exception as e:
        return None
    
def floatify_start_end_times(output_dict_list):
    for d in output_dict_list:
        d['start'] = float(d['start'])
        d['end'] = float(d['end'])
    return output_dict_list


def fetch_broll_description(wordlevel_info,
                            num_vids,
                            url,
                            openaiapi_key,
                            debug=False):

    success = False
    err_msg = ""

    if openaiapi_key == "":
        openaiapi_key = OPENAI_API_KEY

    assert openaiapi_key != "", "Please enter your OPENAI API KEY"

    headers = {
        "content-type": "application/json",
        "Authorization": "Bearer {}".format(openaiapi_key)}

    chunklevelinfo = chunk_level_transcript(wordlevel_info, chunk_size_s=5)
    subtitles = convert_to_srt(chunklevelinfo)

    prompt_prefix = """{}
    
    Given the subtitles of a video, generate very relevant stock video descriptions and search phrase to insert as B-roll video.
    The start and end timestamps of the B-roll videos should perfectly match with the content that is spoken at that time.
    Strictly don't include any exact word or text labels to be depicted in the videos.
    The description should be one sentence long and should be simple and easy to understand.
    The description should represent the caption of the B-roll video and should be simple.
    All of search_phrase should be very relevant to the description and should be very relevant to the Text of Interest.
    The search_phrase should be 1-4 words long.
    the search_phrase should be a string.
    Don't make the timestamps of different videos overlap.
    Leave enough time gap between different B-Roll video appearances so that the original footage is also played as necessary.
    Strictly output only JSON in the output using the format (BE CAREFUL NOT TO MISS ANY COMMAS, QUOTES OR SEMICOLONS ETC)-""".format(subtitles)

    sample = [
        {"description": "...", "search_phrase": "...", "start": "...", "end": "..."},
        {"description": "...", "search_phrase": "...", "start": "...", "end": "..."}
    ]
    prompt = prompt_prefix + json.dumps(sample) + f"""\n
    Be sure to only make {num_vids} jsons. \nJSON:"""

    # Define the payload for the chat model
    messages = [
        {"role": "system", "content": "You are an expert short form video script writer for Instagram Reels and Youtube shorts."},
        {"role": "user", "content": prompt}
    ]

    chatgpt_payload = {
        "model": "gpt-4",
        "messages": messages,
        "temperature": 1.3,
        "max_tokens": 2000,
        "top_p": 1,
        "stop": ["###"]
    }
    while not success:
        success = False
        # Make the request to OpenAI's API
        response = requests.post(url, json=chatgpt_payload, headers=headers)
        response_json = response.json()

        try:
            print("response ",
                  response_json['choices'][0]['message']['content'])
        except:
            print("response ", response_json)
            if 'error' in response_json:
                err_msg = response_json['error']
                return None, err_msg
            success = False
            continue
        # Extract data from the API's response
        try:
            output = json.loads(
                response_json['choices'][0]['message']['content'].strip())
            output = floatify_start_end_times(output)

            if debug:
                print("output: ", output)
            success = validate_KV_pair(output, debug=debug)
            if success:
                print("JSON: ", output)
                success = True
            else:
                print("Could not validate Key-Value pairs in JSON")
                print("Trying again...")
                success = False
                continue
        except Exception as e:
            print("Error in response from OPENAI GPT-4: ", e)

            output = json_corrector(response_json['choices'][0]['message']['content'].strip(),
                                    str(e),
                                    openaiapi_key)
            if output is not None and validate_KV_pair(output, debug=debug):
                print("Corrected JSON: ", output)
                success = True
            else:
                print("Could not correct JSON")
                print("Trying again...")
                success = False
                continue

    return output, err_msg


# %%

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, sc.ndarray):
            return obj.tolist()
        elif isinstance(obj, sc.float32):
            return float(obj)
        elif isinstance(obj, sc.float64):
            return float(obj)
        elif isinstance(obj, sc.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def pipeline(word_level_transcript,
             n_vids_hint,
             n_searches_per_broll=3,
             openaiapi_key=os.getenv("OPENAI_API_KEY"),
             debug=False
             ):
    try:
        # Fetch B-roll descriptions
        broll_descriptions, err_msg = fetch_broll_description(word_level_transcript,
                                                              num_vids=n_vids_hint,
                                                              url=chatgpt_url,
                                                              openaiapi_key=openaiapi_key,
                                                              debug=debug)
        if debug:
            print("B-roll descriptions: ", broll_descriptions)
        if err_msg != "" and broll_descriptions is None:
            return err_msg

        video_informations = []

        # Download videos from Pexels
        for broll_description in broll_descriptions:
            video_dict, err_msg = download_pexels_video([broll_description['search_phrase']],
                                                        a_roll_start_s=broll_description['start'],
                                                        a_roll_end_s=broll_description['end'],
                                                        n_searches_per_phrase=n_searches_per_broll,
                                                        debug=debug)
            if debug:
                print("Video dict: ", video_dict)
                if err_msg != "":
                    print("Error message: ", err_msg)
            if err_msg != "" and video_dict is None:
                return err_msg

            # Rank videos based on the description
            ranked_video = rank_videos(video_dict,
                                       broll_descriptions[0]['description'],
                                       top_K=1)
            video_informations.append(ranked_video[0])

        # ranked_videos.append({"B-roll description": broll_descriptions[0]['description'],
        #                       "Search Phrases": broll_descriptions[0]['search_phrases']})

        return json.loads(json.dumps(video_informations,
                                     cls=NumpyEncoder)
                          )
    except Exception as e:
        err_msg = "Error in pipeline: " + str(e) + "\n" + tb.format_exc()
        print(err_msg)
        return err_msg


# %%
if __name__ == "__main__":
    from example import example_transcript

    context_start_s = 12
    context_end_s = 30
    context_buffer_s = 5
    import time
    t0 = time.time()
    img_info = pipeline(example_transcript,
                        n_vids_hint=6,
                        n_searches_per_broll=2,
                        openaiapi_key=OPENAI_API_KEY,
                        debug=True)
    t1 = time.time()
    print("Time taken: ", t1-t0)
# %%
