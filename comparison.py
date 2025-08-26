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

def get_all_frames(video_paths):
    return [get_keyframe(p) for p in video_paths]

def create_batch(device, preprocess, frames):
    pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    preprocessed = [preprocess(img) for img in pil_images]
    return torch.stack(preprocessed).to(device)   # shape: (N, 3, 224, 224)

def get_embeddings(model, batch):
    with torch.no_grad():
        embeddings = model.encode_image(batch)
        return embeddings / embeddings.norm(dim=-1, keepdim=True)  # normalize

def create_similarity_matrix(device, model, preprocess, video_paths):
    frames = get_all_frames(video_paths)
    batch = create_batch(device, preprocess, frames)
    embeddings = get_embeddings(model, batch)
    similarity_matrix = embeddings @ embeddings.T
    return similarity_matrix.cpu().numpy()

def get_matrix_row_by_indicies_sorted(matrix, row_index):
    row = matrix[row_index]
    return row.argsort()[::-1]

def get_matrix_row_sorted(matrix, row_index):
    row = matrix[row_index]
    return np.sort(row)[::-1]

def get_rows_that_contain_top_score(top_score, score_list):
    return [i for i, x in enumerate(score_list) if x == top_score]

def order_top_two_clips(matrix, top_clips):
    second_highest_score_of_first_clip = -1
    final_clip_index_order = []

    for clip in top_clips:
        sorted_row = get_matrix_row_sorted(matrix, clip)
        if sorted_row[2] > second_highest_score_of_first_clip:
            second_highest_score_of_first_clip = sorted_row[2]
            final_clip_index_order.append(clip)
        else:
            final_clip_index_order.insert(0, clip)

    return final_clip_index_order

def get_first_two_clips(video_count, similarity_matrix):
    score_list = []

    for i in range(video_count):
        sorted_row = get_matrix_row_sorted(similarity_matrix, i)
        score_list.append(sorted_row[1].item())
    top_score = max(score_list)
    top_two_clips = get_rows_that_contain_top_score(top_score, score_list)

    return order_top_two_clips(similarity_matrix, top_two_clips)

def add_used_scores(used_scores, matrix, row_index):
    sorted_row = get_matrix_row_sorted(matrix, row_index)
    for score in sorted_row[1:]:
        used_scores.append(score)
    return used_scores

def get_score(matrix, row_index):
    sorted_row = get_matrix_row_sorted(matrix, row_index)
    return sorted_row[2]

def get_scores(video_paths):

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    similarity_matrix = create_similarity_matrix(device, model, preprocess, video_paths)
    print(similarity_matrix)

    used_scores = []

    final_clips = get_first_two_clips(video_paths, similarity_matrix)

    used_scores = add_used_scores(used_scores, similarity_matrix, final_clips[0])
    used_scores = add_used_scores(used_scores, similarity_matrix, final_clips[1])

    next_score = get_score()

    next_score = sims[sorted_idx[2]].item()

    sims = similarity_matrix[final_clips[0]]
    sorted_idx = sims.argsort()[::-1]
    for j in sorted_idx[1:]:
        used_scores.append(sims[j].item())

    used_scores.append(next_score)

    second_final_clip_index_order = []

    for x in range(len(video_paths) - 2):
        if(len(second_final_clip_index_order) >= len(video_paths) - 2): break
        for i, path in enumerate(video_paths):
            if(len(second_final_clip_index_order) >= len(video_paths) - 2): break
            if i in final_clips: continue
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

    final_list = final_clips + second_final_clip_index_order

    print(final_list)

    return final_list