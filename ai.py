import cv2
import torch
import clip
import os
import math
from PIL import Image
import numpy as np
from moviepy import VideoFileClip
import torch.nn.functional as F



def get_keyframe(video_path):
    clip = VideoFileClip(video_path)
    frame = clip.get_frame(clip.duration / 2)  # middle frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert to OpenCV BGR
    return frame

def get_scores(video_paths):

    # Load model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    frames = [get_keyframe(p) for p in video_paths]



    pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    preprocessed = [preprocess(img) for img in pil_images]
    batch = torch.stack(preprocessed).to(device)   # shape: (N, 3, 224, 224)

    with torch.no_grad():
        embeddings = model.encode_image(batch)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # normalize


    similarity_matrix = embeddings @ embeddings.T   # (N x N)
    similarity_matrix = similarity_matrix.cpu().numpy()

    print(similarity_matrix)

    all_scores = []
    used_scores = []

    for i, path in enumerate(video_paths):
        sims = similarity_matrix[i]
        sorted_idx = sims.argsort()[::-1]
        for j in sorted_idx[1:]:
            all_scores.append(sims[j].item())

    final_clip_index_order = []

    score_list = []
    seconday_score_list = []

    for i, path in enumerate(video_paths):
        sims = similarity_matrix[i]
        sorted_idx = sims.argsort()[::-1]
        score_list.append(sims[sorted_idx[1]].item())
    top_score = max(score_list)
    used_scores.append(top_score)
    top_clip_list = [i for i, x in enumerate(score_list) if x == top_score]


    for i in range(len(top_clip_list)):
        sims = similarity_matrix[top_clip_list[i]]
        sorted_idx = sims.argsort()[::-1]
        seconday_score_list.append(sims[sorted_idx[2]].item())

    bottom_score = min(seconday_score_list)
    other_score = max(seconday_score_list)
    min_index = seconday_score_list.index(bottom_score)
    other_index = seconday_score_list.index(other_score)

    first_clip_index = top_clip_list[min_index]
    second_clip_index = top_clip_list[other_index]
    final_clip_index_order.append(first_clip_index)
    final_clip_index_order.append(second_clip_index)


    sims = similarity_matrix[final_clip_index_order[-1]]
    sorted_idx = sims.argsort()[::-1]
    for j in sorted_idx[1:]:
        used_scores.append(sims[j].item())
    next_score = sims[sorted_idx[2]].item()

    sims = similarity_matrix[final_clip_index_order[0]]
    sorted_idx = sims.argsort()[::-1]
    for j in sorted_idx[1:]:
        used_scores.append(sims[j].item())

    used_scores.append(next_score)


    second_final_clip_index_order = []

    for x in range(len(video_paths) - 2):
        if(len(second_final_clip_index_order) >= len(video_paths) - 2): break
        for i, path in enumerate(video_paths):
            if(len(second_final_clip_index_order) >= len(video_paths) - 2): break
            if i in final_clip_index_order: continue
            sims = similarity_matrix[i]
            sorted_idx = sims.argsort()[::-1]
            temp_list = []
            for j in sorted_idx[1:]:
                temp_list.append(sims[j].item())
            if next_score in temp_list:
                second_final_clip_index_order.append(i)
                set2 = set(used_scores)
                unique_elements = [item for item in temp_list if item not in set2]

                if unique_elements:  # Check if the list is not empty
                    next_score = max(unique_elements)
                    used_scores.append(next_score)
                    for j in range(len(temp_list)):
                        used_scores.append(temp_list[j])

    final_list = final_clip_index_order + second_final_clip_index_order

    return final_list