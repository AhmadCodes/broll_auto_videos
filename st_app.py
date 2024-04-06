import os
# curr_dir = os.path.dirname(os.path.abspath(__file__))
# lock_file = os.path.join(curr_dir,'streamlit.lock')
from app.main_app import pipeline



import streamlit as st
st.session_state["RERUN"] = 0
RERUN=st.session_state["RERUN"]
st.set_page_config(page_title="B-Roll Images", page_icon=":camera:")
import traceback

# Display the Streamlit app title and description
st.title("B-Roll Images")
st.write("Generate B-roll Images and insert to video.")

# @st.cache_resource(max_entries=1, show_spinner = "Initializing... This may take a while.")
# def initializer(rerun):
#     # Load your machine learning model here
#     if "pipeline" in st.session_state:
#         # global PIPELINE
#         del st.session_state["pipeline"]
#         print("Pipeline deleted")
#     # with st.spinner("Initializing... This may take a while."):
#     from app.main_app import pipeline
#     st.session_state["pipeline"] = pipeline
#         # return pipeline

# # Load the model
# # PIPELINE = initializer(RERUN)
# initializer(RERUN)

# def clear_cache():
#     # initializer.clear()
#     st.cache_resource.clear()
#     # del PIPLINE
#     if "RERUN" not in st.session_state:
#         st.session_state["RERUN"] = 0
    
#     # if "PIPELINE" in globals():
#     #     del PIPELINE
#     #     print("Pipeline deleted")
    
#     st.session_state["RERUN"] +=1
#     RERUN=st.session_state["RERUN"]
#     # PIPELINE = initializer(RERUN)
#     initializer(RERUN)
    

# st.sidebar.button("Clear Cache and Re-Initialize",on_click=clear_cache)


# Model options and SD models
model_options = ["tiny", "base", "small", "medium",]# "large-v1", "large-v2"]
sd_models = [
    
    "Lykon/dreamshaper-xl-turbo",
    "lykon/dreamshaper-8-lcm",
    "stabilityai/sdxl-turbo",
]

# Streamlit widgets
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
import os
temp_file_path = None
if uploaded_file is not None:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(curr_dir, "assets/")
    # Create a temporary directory
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    # Define the file path in the temporary directory
    temp_file_path = temp_dir + f"/{uploaded_file.name}"
    
    # Write the contents of the uploaded file to the temporary directory
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    moved_file_path = os.path.join(curr_dir, "assets/", uploaded_file.name)
    # os.rename(temp_file_path, moved_file_path)
    moved_file_path = temp_file_path
    # st.write("File uploaded successfully")
    st.success("File uploaded successfully")
            
            
apikey = st.text_input("Enter your OPENAI API KEY")
model_type = st.selectbox("Transcription Model", model_options)
sd_model = st.selectbox("Image Generation Model", sd_models)
steps = st.slider("BRoll Image Quality (FAST <--> DETAILED)", 4, 50, 30)



# Check if the lock file exists


if st.button("Generate Video"):
    progress_bar = st.progress(0, "Starting...")
    with st.spinner('Generating video...'):
        if temp_file_path is None:
            st.write("Please upload a video file")
        else:
            finalvideopath = ""
            try:
                finalvideopath, err_msg = pipeline(
                    video_file=moved_file_path,
                    broll_image_steps=steps,
                    transcription_model=model_type,
                    SD_model=sd_model,
                    openaiapi_key=apikey,
                    progress=progress_bar
                )
            
            except Exception as e:
                err_msg = str(e)
                traceback.print_exc()
    
    if err_msg == "":
        st.success("Video generated successfully!")
        st.video(data=finalvideopath, format="video/mp4", start_time=0)
    else:
        st.error(f"Error: {err_msg}")
    
    # os.remove(lock_file)


