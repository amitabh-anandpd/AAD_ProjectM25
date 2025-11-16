from PIL import Image, ImageDraw, ImageFont

def array_to_image(grid, cell_size=80, filename="output.png"):
    rows = len(grid)
    cols = len(grid[0])

    width = cols * cell_size
    height = rows * cell_size

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Load a font safely
    try:
        font = ImageFont.truetype("arial.ttf", int(cell_size * 0.5))
    except:
        font = ImageFont.load_default()

    for r in range(rows):
        for c in range(cols):
            x1 = c * cell_size
            y1 = r * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            # Draw cell border
            draw.rectangle([x1, y1, x2, y2], outline="black", width=2)

            text = str(grid[r][c])

            # Use textbbox instead of textsize
            bbox = draw.textbbox((0,0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            draw.text(
                (x1 + (cell_size - w) / 2, y1 + (cell_size - h) / 2),
                text,
                fill="black",
                font=font
            )

    img.save(filename)
    print("Saved:", filename)
    return img


# Test
grid = [
    [3, 1, 4, 5, 6, 7]
]
array_to_image(grid)
