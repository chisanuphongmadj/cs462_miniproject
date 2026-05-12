from PIL import Image
import numpy as np


INK_THRESHOLD = 245
PADDING_RATIO = 0.20


def crop_and_center_image(image, image_size, ink_threshold=INK_THRESHOLD, padding_ratio=PADDING_RATIO):
    """Crop around dark ink, center it on a square canvas, then resize."""
    image = image.convert("L")
    array = np.asarray(image)
    mask = array < ink_threshold

    if not mask.any():
        return image.resize((image_size, image_size), Image.Resampling.LANCZOS)

    ys, xs = np.where(mask)
    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1

    crop = image.crop((left, top, right, bottom))
    width, height = crop.size
    content_size = max(width, height)
    padding = max(2, int(round(content_size * padding_ratio)))
    square_size = content_size + padding * 2

    square = Image.new("L", (square_size, square_size), 255)
    paste_x = (square_size - width) // 2
    paste_y = (square_size - height) // 2
    square.paste(crop, (paste_x, paste_y))
    return square.resize((image_size, image_size), Image.Resampling.LANCZOS)


def split_digit_images(image, image_size, ink_threshold=INK_THRESHOLD):
    """Split a two-digit canvas into left/right normalized digit crops."""
    image = image.convert("L")
    array = np.asarray(image)
    mask = array < ink_threshold

    if not mask.any():
        blank = Image.new("L", (image_size, image_size), 255)
        return blank, blank

    ys, xs = np.where(mask)
    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1

    content = image.crop((left, top, right, bottom))
    content_mask = mask[top:bottom, left:right]
    height, width = content_mask.shape

    if width <= 2:
        split_x = max(1, width // 2)
    else:
        counts = content_mask.sum(axis=0).astype(np.float32)
        window = max(3, width // 24)
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window, dtype=np.float32) / window
        smoothed = np.convolve(counts, kernel, mode="same")

        search_left = max(1, int(width * 0.28))
        search_right = min(width - 1, int(width * 0.76))
        central = smoothed[search_left:search_right]

        if central.size:
            low_threshold = max(1.0, float(smoothed.max()) * 0.12)
            low_columns = np.where(central <= low_threshold)[0] + search_left
            if low_columns.size:
                groups = np.split(low_columns, np.where(np.diff(low_columns) != 1)[0] + 1)
                best_group = max(groups, key=len)
                split_x = int(round((int(best_group[0]) + int(best_group[-1])) / 2))
            else:
                split_x = int(search_left + int(np.argmin(central)))
        else:
            split_x = width // 2

        split_x = max(1, min(width - 1, split_x))

    left_digit = content.crop((0, 0, split_x, height))
    right_digit = content.crop((split_x, 0, width, height))
    return crop_and_center_image(left_digit, image_size), crop_and_center_image(right_digit, image_size)


def random_affine_on_canvas(image, rng, max_angle=8, max_translate=6, min_scale=0.88, max_scale=1.08):
    """Apply small post-normalization variation on a white canvas."""
    image = image.convert("L")
    angle = rng.uniform(-max_angle, max_angle)
    image = image.rotate(angle, fillcolor=255)

    scale = rng.uniform(min_scale, max_scale)
    scaled_size = max(1, int(round(image.size[0] * scale)))
    scaled = image.resize((scaled_size, scaled_size), Image.Resampling.LANCZOS)

    canvas = Image.new("L", image.size, 255)
    translate_x = rng.randint(-max_translate, max_translate)
    translate_y = rng.randint(-max_translate, max_translate)
    paste_x = (image.size[0] - scaled_size) // 2 + translate_x
    paste_y = (image.size[1] - scaled_size) // 2 + translate_y
    canvas.paste(scaled, (paste_x, paste_y))
    return canvas
