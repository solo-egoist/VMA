import os
import random
import subprocess
import tempfile
from comparison import get_scores
from moviepy import VideoFileClip, concatenate_videoclips # type: ignore

directory_path = "F:\Recordings\JJK\Season 2\Episode 4 Cleaned"
all_entries = os.listdir(directory_path)
clip_count = 4
max_clip_length = 9

def dedupe_clip(input_path):
    # Create a temporary file for the deduped video
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_file.close()  # close so ffmpeg can write to it

    # ffmpeg mpdecimate command
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "mpdecimate,setpts=N/FRAME_RATE/TB",  # drop dupes & fix timestamps
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",  # strip audio if you don’t need it, remove this if you want audio
        tmp_file.name
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return tmp_file.name



sampled_episodes = []

for x in range(clip_count):
    current_clip_length = 99
    while current_clip_length >= max_clip_length + 1:
        current_clip = random.sample(all_entries, 1)[0]
        file_path = os.path.join(directory_path, current_clip)
        clip = VideoFileClip(file_path)
        current_clip_length = clip.duration
        print(current_clip_length)
    
    dedupedClip = dedupe_clip(file_path)
    sampled_episodes.append(dedupedClip)


clips = []


for sample in sampled_episodes:
    clips.append(VideoFileClip(sample))

clip_scores = get_scores(sampled_episodes)

final_clip_list = []

for i in range(len(clip_scores)):
    idx = clip_scores[i]
    final_clip_list.append(clips[idx])


final_clip = concatenate_videoclips(final_clip_list, method="chain")

final_clip.write_videofile("F:\Recordings\JJK\Season 2\Episode 13 Cleaned\output.mp4")

for clip in clips:
    clip.close()
final_clip.close()