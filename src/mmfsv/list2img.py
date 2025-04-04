# list2img.py
import math
import sys

def deal_list(mid_sign_list):

    import numpy as np

    n_rows = len(mid_sign_list)
    n_cols = 11
    mid_sign_img = np.zeros((n_rows, n_cols), dtype=np.float32)

    show_progress = True
    last_print = -1

    for i, mid_sign in enumerate(mid_sign_list):
        length_all = len(mid_sign)
        mid_sign_img[i, 0] = length_all

        if length_all <= 1:
            continue

        mid_sign_sub = sorted(mid_sign[1:])
        length_sub = len(mid_sign_sub)

        mid_sign_img[i, 1] = mid_sign_sub[0]
        mid_sign_img[i, 2] = mid_sign_sub[length_sub // 4]
        mid_sign_img[i, 3] = mid_sign_sub[length_sub // 2]
        mid_sign_img[i, 4] = mid_sign_sub[length_sub * 3 // 4]
        mid_sign_img[i, 5] = mid_sign_sub[-1]
        power_sum = sum(x * x for x in mid_sign_sub)
        rms_val = math.sqrt(power_sum / length_sub)
        mid_sign_img[i, 6] = rms_val
        h_sum = sum((1.0 / x) for x in mid_sign_sub)
        hm_val = length_sub / h_sum
        mid_sign_img[i, 7] = hm_val
        mean_val = sum(mid_sign_sub) / length_sub
        mid_sign_img[i, 8] = mean_val
        accum = sum((x - mean_val) ** 2 for x in mid_sign_sub)
        std_val = math.sqrt(accum / length_sub)
        mid_sign_img[i, 9] = std_val
        if mean_val != 0:
            mid_sign_img[i, 10] = std_val / mean_val
        else:
            mid_sign_img[i, 10] = 0.0

        if show_progress:
            progress = int((i + 1) * 100 / n_rows)
            if (i % 1000 == 0 or progress == 100) and progress != last_print:
                bar_width = 50
                filled_len = bar_width * progress // 100
                bar_str = '=' * filled_len + '>' + ' ' * (bar_width - filled_len - 1)
                sys.stdout.write(f" > [{bar_str}] {progress:.1f}% ({i+1} / {n_rows})\r")
                sys.stdout.flush()
                last_print = progress

    if show_progress:
        print()

    return mid_sign_img
