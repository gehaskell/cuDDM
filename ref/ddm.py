import numpy as np
import logging
import random
import cv2
import matplotlib.pyplot as plt
import time 
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

SINGLE = np.dtype("single")


def load_to_buffer(capture, frames_left, frame_count, buffer, buffer_offset, width=None, height=None):
    buffer_size = buffer.shape[0]

    if not width:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if not height:
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)/2 - width/2)
    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)/2 - height/2)

    end_of_video = False
    for i in range(frame_count):
        # read frame
        ret, frame = cap.read()
        if ret and frames_left > 0:
            index = (buffer_offset + i) % buffer_size
            frame = frame[w: w+width, h: h+height]
            buffer[index] = np.average(frame, axis=2, weights=[.3, .6, .11])
            frames_left -= 1
        else:
            end_of_video = True
            frame_count = i
            capture.release()
            break

    logging.info(f"Loaded {frame_count} frames into buffer {buffer_offset} - {buffer_offset+frame_count-1}")
    if end_of_video: logging.info("End of video.")

    return end_of_video, frame_count


def chunk_analysis(buffer, buffer_offset, chunk_size, fft_mag_accum, ccc_vector, tau_vector):
    """
    Handles DDM analysis on a single

    :param buffer: main data array [#frame, width, height]
    :param buffer_offset: frame offset for analysis
    :param chunk_size: #frame for analysis chunk
    :param fft_mag_accum: Accumulated fourier-transform data array [tau_index, width, height]
    :param ccc_vector: Number of frames that have contributed to fft_mag_accum
    :param tau_vector: Vector containing tau values
    """
    if chunk_size <= 0:
        return


    buffer_size = buffer.shape[0]

    pixel_count = buffer.shape[1] * buffer.shape[2]
    norm_factor = 1 / pixel_count

    logging.info(f"Analysing buffer frames {buffer_offset}-{buffer_offset+chunk_size-1}")

    for tau_idx, tau in enumerate(tau_vector):
        # Currently only take of each tau value per chunk
        for x in range(20):
            index_1 = buffer_offset + random.randint(0, chunk_size - 1 - tau)  # random is likely a bottle neck

            temp_diff = buffer[index_1 % buffer_size] - buffer[(index_1 + tau) % buffer_size]
            fft_temp_diff = np.fft.fft2(temp_diff) * norm_factor
            
            #plt.pcolor(temp_diff)
            #plt.show()

            fft_mag_accum[tau_idx] += np.abs(fft_temp_diff) * 2

            # plt.pcolor(fft_mag_accum[tau_idx])
            # plt.show()

            ccc_vector[tau_idx] += 1


def gen_radius_masks(q_num, width, height):
    radius_mask = np.zeros((q_num, width, height))
    radius_mask_px_count = np.zeros(q_num)

    q_vector = np.linspace(1, width/7, num=q_num)

    for q_index in range(q_num):
        logging.info(f"Radial mask {q_index}")
        for x in range(width):
            for y in range(height):
                r = np.linalg.norm([x - width / 2, y - height / 2])
                # print(r, q[q_index])
                if 1 <= r / q_vector[q_index] <= 1.2:
                    radius_mask[q_index, x, y] = 1
                    radius_mask_px_count[q_index] += 1
        #plt.pcolor(radius_mask[q_index])
        #plt.show()
        radius_mask[q_index] = np.fft.fftshift(radius_mask[q_index])
    # print(np.array2string(q_vector, separator=', '))
    # print(np.array2string(q_vector, separator=', '))
    # print(q_vector.size)

    return radius_mask, q_vector, radius_mask_px_count


def analyse_fft_mag_accum(fft_mag_accum, ccc_vector, radius_masks, q_vector, radius_mask_px_count):
    tau_count = ccc_vector.size
    q_count = q_vector.size

    iqtau = np.zeros((q_count, tau_count))

    for tau_index in range(tau_count):
        print(ccc_vector[tau_index])
        fft_mag_accum[tau_index] = fft_mag_accum[tau_index] / ccc_vector[tau_index]

        for q_index in range(q_count):
            if radius_mask_px_count[q_index] == 0:
                iqtau[q_index, tau_index] = np.NaN
            else:
                iqtau[q_index, tau_index] = np.sum(radius_masks[q_index] * fft_mag_accum[tau_index]) \
                                            / radius_mask_px_count[q_index]

    return q_vector, iqtau


def ddm_circ(capture, video_width, video_height, tau_vector, frames):
    buffer_size = 120
    chunk_size = 30

    analysis_offset = 0
    load_in_offset = 0
    q_count = 20

    t1 = time.time()

    buffer = np.zeros((buffer_size, video_width, video_height), SINGLE)
    fft_mag_accum = np.zeros((tau_vector.size, video_width, video_height), SINGLE)
    ccc_vector = np.zeros(tau_vector.size)

    # Load in two video chunks
    finished, gen_chunk_size = load_to_buffer(capture, frames, chunk_size, buffer, load_in_offset, video_width, video_height)
    frames -= chunk_size
    load_in_offset = (load_in_offset + gen_chunk_size) % buffer_size

    while not finished:
        chunk_analysis(buffer, analysis_offset, gen_chunk_size, fft_mag_accum, ccc_vector, tau_vector)
        frames -= chunk_size
        analysis_offset = (analysis_offset + gen_chunk_size) % buffer_size

        finished, gen_chunk_size = load_to_buffer(capture, frames, chunk_size, buffer, load_in_offset, video_width, video_height)
        load_in_offset = (load_in_offset + gen_chunk_size) % buffer_size

    chunk_analysis(buffer, analysis_offset, gen_chunk_size, fft_mag_accum, ccc_vector, tau_vector)
    logging.info("Initial analysis complete.")
    t2 = time.time()
    print(t2-t1)

    # Analyse
    radius_masks, q_vector, radius_mask_px_count = gen_radius_masks(q_count, video_width, video_height)
    logging.info("Got radius masks")

    out = analyse_fft_mag_accum(fft_mag_accum, ccc_vector, radius_masks, q_vector, radius_mask_px_count)
    logging.info("Full analysis complete.")
    print(out)
    return out

if __name__ == '__main__':
    # Playing video from file:
    cap = cv2.VideoCapture("/home/ghaskell/projects_Git/cuDDM/data/colloid_0.5um_vid.mp4")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # note this isn't actually super accurate

    #over write
    width = 1024
    height = 1024
    frames = 90


    size = min(width, height) # divide 4 to reduce analysed area - remove for real implementation
    size = int(2 ** np.floor(np.log2(size)))

    print(f"Width: {width}, Height: {height}, Frames: {frames}, Analysed Size: {size}")

    tau_vec = np.array(range(2, 15))

    q_vector, iqtau = ddm_circ(cap, size, size, tau_vec, frames)
    q_vector = q_vector[1:]
    iqtau = iqtau[1:,:]

    # for tau in range(1, iqtau.shape[1]):
    #     # plotting the points
    #     plt.plot(q_vector, iqtau[:, tau])
    #     plt.xlabel("qs")
    # plt.show()

    for qi in range(iqtau.shape[0]):
        # plotting the points
        plt.plot(tau_vec, iqtau[qi], label=f"q = {q_vector[qi]}")
        plt.xlabel("Tau")
        plt.legend(loc="upper left")
    plt.show()

