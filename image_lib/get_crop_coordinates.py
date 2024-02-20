def get_crop_coordinates(face, h, w):
    left, top, right, bottom = map(int, (face.left(), face.top(), face.right(), face.bottom()))

    head_top = max(0, top - (bottom - top) * 0.5)
    head_bottom = min(h, bottom + (bottom - top) * 0.7)
    head_left = max(0, left - (right - left) * 0.25)
    head_right = min(w, right + (right - left) * 0.25)

    w1, h1 = head_right - head_left, head_bottom - head_top
    aspect = w1 / h1

    if aspect > 178 / 218:
        head_bottom = min(h, int(head_top + w1 / 178 * 218))
        excess = max(0, head_bottom - h)
        head_top, head_bottom = max(0, head_top - excess), min(h, head_bottom - excess)
        compress = int(excess * 178 / 218)
        head_left, head_right = head_left + compress / 2, head_right - compress / 2
    else:
        new_right = int(head_left + h1 / 218 * 178)
        extension = max(0, new_right - head_right)
        head_right += extension / 2
        head_left -= extension / 2
        excess = max(0, head_right - w)
        head_left, head_right = max(0, head_left - excess), min(w, head_right - excess)
        compress = int(excess * 218 / 178)
        head_top, head_bottom = head_top + compress / 2, head_bottom - compress / 2

    return int(head_top), int(head_left), int(head_bottom), int(head_right)
