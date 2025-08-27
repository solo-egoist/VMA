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
    return np.sort(row)[::-1][1:]

def get_rows_that_contain_top_score(top_score, score_list):
    return [i for i, x in enumerate(score_list) if x == top_score]

def order_top_two_clips(matrix, top_clips):
    second_highest_score_of_first_clip = -1
    final_clip_index_order = []

    for clip in top_clips:
        sorted_row = get_matrix_row_sorted(matrix, clip)
        if sorted_row[1] > second_highest_score_of_first_clip:
            second_highest_score_of_first_clip = sorted_row[1]
            final_clip_index_order.append(clip)
        else:
            final_clip_index_order.insert(0, clip)

    return final_clip_index_order

def get_first_two_clips(video_count, similarity_matrix):
    score_list = []

    for i in range(video_count):
        sorted_row = get_matrix_row_sorted(similarity_matrix, i)
        score_list.append(sorted_row[0].item())
    top_score = max(score_list)
    top_two_clips = get_rows_that_contain_top_score(top_score, score_list)

    return order_top_two_clips(similarity_matrix, top_two_clips)

def add_used_scores(used_scores, matrix, row_index):
    sorted_row = get_matrix_row_sorted(matrix, row_index)
    used_scores.extend(sorted_row)
    return used_scores

def get_score(matrix, row_index):
    sorted_row = get_matrix_row_sorted(matrix, row_index)
    return sorted_row[1]

def row_is_in_list(row_index, clip_list):
    return row_index in clip_list

def all_clips_are_sorted(final_clips_list, clip_count):
    return len(final_clips_list) == clip_count

def organize_remaining_clips(final_clips, clip_count, used_scores, matrix, next_score):
    index = -1
    while all_clips_are_sorted(final_clips, clip_count) == False:
        if index >= clip_count - 1: index = -1
        index += 1

        if row_is_in_list(index, final_clips): continue
        sorted_row = get_matrix_row_sorted(matrix, index)

        if next_score not in sorted_row: continue

        final_clips.append(index)
        used_scores.append(next_score)
        set2 = set(used_scores)
        unique_elements = [item for item in sorted_row if item not in set2]

        if unique_elements == []: continue

        next_score = max(unique_elements)
        used_scores.extend(sorted_row)

    return final_clips

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

    next_score = get_score(similarity_matrix, 1)

    final_list = organize_remaining_clips(final_clips, len(video_paths), used_scores, similarity_matrix, next_score)

    print(final_list)

    return final_list