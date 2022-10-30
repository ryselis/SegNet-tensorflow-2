import os
import random

import numpy

from BinaryMask_pb2 import BinaryMaskSequence
from config import DATA_SHAPE, DATA_PADDING, DATA_FILE_LIMIT, BATCH_SIZE, base_dir


def get_dataset_generator():
    all_files = [os.path.join(dirpath, f)
                 for dirpath, dirnames, filenames in os.walk(base_dir)
                 for f in filenames
                 if f.endswith('.mdf')]
    random.seed(2)
    random.shuffle(all_files)
    if DATA_FILE_LIMIT is not None:
        all_files = all_files[:DATA_FILE_LIMIT]
    train_data_size = int(len(all_files) * 0.8)

    file_batch_size = 20
    return (lambda: generate_dataset(all_files[:train_data_size], file_batch_size)), \
           (lambda: generate_dataset(all_files[train_data_size:], file_batch_size // 5))


def generate_dataset(file_list, file_batch_size):
    random.shuffle(file_list)

    file_batch_size = max(file_batch_size, 1)

    for start_index in range(0, len(file_list), file_batch_size):
        file_sublist = file_list[start_index:start_index + file_batch_size]
        yield from _generate_frames(file_sublist)


def _generate_frames(file_sublist):
    frames = []
    for mask_file in file_sublist:
        with open(mask_file, 'rb') as f:
            binary_masks = BinaryMaskSequence()
            binary_masks.ParseFromString(f.read())
        frames.extend(binary_masks.frames)
    random.shuffle(frames)
    frame_pairs = [frames[index:index + BATCH_SIZE] for index in range(0, len(frames), BATCH_SIZE)]
    for frame_pair in frame_pairs:
        x_values = numpy.array([
            numpy.pad(
                numpy.array(frame.depthFrame.points, dtype='float32').reshape((DATA_SHAPE[0], DATA_SHAPE[1], 1)),
                pad_width=DATA_PADDING
            ) for frame in frame_pair
        ])

        x_values /= numpy.max(x_values)
        y_values = numpy.array([numpy.pad(
            numpy.array(frame.binaryMask.points).reshape((DATA_SHAPE[0], DATA_SHAPE[1])),
            pad_width=DATA_PADDING[:2])
            for frame in frame_pair
        ])
        if numpy.max(y_values) == 1:
            yield x_values, y_values
