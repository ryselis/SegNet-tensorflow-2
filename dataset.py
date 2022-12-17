import os
import random

import numpy
from scipy.ndimage import zoom

from BinaryMask_pb2 import BinaryMaskSequence
from config import DATA_SHAPE, DATA_FILE_LIMIT, BATCH_SIZE, base_dir


def get_dataset_generator(batch_size=None, skip_frames=None):
    all_files = get_all_files()
    random.seed(2)
    random.shuffle(all_files)
    if DATA_FILE_LIMIT is not None:
        all_files = all_files[:DATA_FILE_LIMIT]
    train_data_size = int(len(all_files) * 0.8)

    file_batch_size = 20
    return (lambda: generate_dataset(all_files[:train_data_size], file_batch_size, batch_size=batch_size,
                                     skip_frames=skip_frames)), \
           (lambda: generate_dataset(all_files[train_data_size:], file_batch_size // 5, batch_size=batch_size,
                                     loop=False))


def get_all_files() -> list[str]:
    all_files = [os.path.join(dirpath, f)
                 for dirpath, dirnames, filenames in os.walk(base_dir)
                 for f in filenames
                 if f.endswith('.mdf')]
    return all_files


def get_full_data_generator(batch_size=None, simple_samples=None, complex_samples=None):
    all_files = get_all_files()
    files_by_top_dir = group_files_by_top_dir(all_files)
    if simple_samples is None:
        simple_samples = len(files_by_top_dir['GenderResearch'])
    if complex_samples is None:
        complex_samples = len(files_by_top_dir['TripleKinectResearch'])
    yield from generate_frames_from_file_list(files_by_top_dir['GenderResearch'][:simple_samples],
                                              batch_size=batch_size)
    yield from generate_frames_from_file_list(files_by_top_dir['TripleKinectResearch'][:complex_samples],
                                              batch_size=batch_size)

def generate_frames_from_file_list(all_files, batch_size=None):
    for mask_file in all_files:
        binary_masks = read_file_frames(mask_file)
        yield mask_file, generate_frame_batches(binary_masks.frames, batch_size, skip_frames=0, total_frames=0)

def group_files_by_top_dir(files: list[str]) -> dict[str, list[str]]:
    groups = {}
    for f in files:
        rel_name = f.replace(f'{base_dir}/', '')
        dir_name = rel_name.split('/', 1)[0]
        groups.setdefault(dir_name, [])
        groups[dir_name].append(f)
    return groups


def generate_dataset(file_list, file_batch_size, batch_size=None, skip_frames=None, loop=True):
    current_frame = 0
    skip_frames = skip_frames or 0
    while True:
        random.shuffle(file_list)

        file_batch_size = max(file_batch_size, 1)

        for start_index in range(0, len(file_list), file_batch_size):
            file_sublist = file_list[start_index:start_index + file_batch_size]
            frames_left_to_skip = max(skip_frames - current_frame, 0)
            for frame_data in _generate_frames(file_sublist, batch_size=batch_size, skip_frames=frames_left_to_skip):
                if frame_data is not None:
                    yield frame_data
                current_frame += 1
        if not loop:
            break


def _generate_frames(file_sublist, batch_size=None, skip_frames=0):
    total_frames = 0
    frames = []
    for mask_file in file_sublist:
        binary_masks = read_file_frames(mask_file)
        frames.extend(binary_masks.frames)
    random.shuffle(frames)
    size = batch_size or BATCH_SIZE
    yield from generate_frame_batches(frames, size, skip_frames, total_frames)


def read_file_frames(mask_file):
    with open(mask_file, 'rb') as f:
        binary_masks = BinaryMaskSequence()
        binary_masks.ParseFromString(f.read())
    return binary_masks


def generate_frame_batches(frames, size, skip_frames, total_frames):
    frame_pairs = [frames[index:index + size] for index in range(0, len(frames), size)]
    for frame_pair in frame_pairs:
        if len(frame_pair) != size:
            continue
        total_frames += 1
        if total_frames < skip_frames:
            yield None
            continue

        # transform x
        x_reshaped = [numpy.array(frame.depthFrame.points, dtype='float32').reshape((DATA_SHAPE[0], DATA_SHAPE[1], 1))
                      for frame in frame_pair]
        x_padded = numpy.array(x_reshaped)
        scale_x = 360 / x_padded.shape[1]
        scale_y = 480 / x_padded.shape[2]
        zoom_values = [1, scale_x, scale_y, 1]
        x_zoomed = zoom(input=x_padded, zoom=zoom_values, mode='nearest', prefilter=False)
        x_zoomed /= numpy.max(x_zoomed) * 255
        x_with_3_dims = numpy.broadcast_to(x_zoomed, shape=[x_zoomed.shape[0], x_zoomed.shape[1], x_zoomed.shape[2], 3])
        # transform y
        # 0 -> 12, 1 -> 10  y = 12 - 2 * x
        original_y_values = [numpy.array(frame.binaryMask.points) for frame in frame_pair]
        # adapted_y_values = [-2 * numpy.array(value) + 12 for value in original_y_values]
        y_values = numpy.array([
            frame.reshape((DATA_SHAPE[0], DATA_SHAPE[1], 1))
            for frame in original_y_values
        ])
        y_values = zoom(input=y_values, zoom=zoom_values, mode='nearest', prefilter=False)
        # y_values = -2 * y_values + 12
        if 1 in y_values:
            yield x_with_3_dims, y_values

