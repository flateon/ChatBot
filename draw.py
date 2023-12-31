from PIL import Image
import rich
from rich.console import Console
import os
import time


def to_string(img: Image, dest_width: int, unicode: bool = True) -> str:
    img_width, img_height = img.size
    scale = img_width / dest_width
    dest_height = int(img_height / scale)
    dest_height = dest_height + 1 if dest_height % 2 != 0 else dest_height
    img = img.resize((dest_width, dest_height))
    output = ""
    
    for y in range(0, dest_height, 2):
        for x in range(dest_width):
            if unicode:
                r1, g1, b1 = img.getpixel((x, y))
                r2, g2, b2 = img.getpixel((x, y + 1))
                output = output + f"[rgb({r1},{g1},{b1}) on rgb({r2},{g2},{b2})]▀[/]"
            else:
                r, g, b = img.getpixel((x, y))
                output = output + f"[on rgb({r},{g},{b})] [/]"
        
        output = output + "\n"
    
    return output


if __name__ == '__main__':
    # rich.print("[on rgb(255,0,0)]▀[/]")
    
    folder = './icon_gif/'
    img_strs = []
    for f in os.listdir(folder):
        img = Image.open(os.path.join(folder, f))
        s = to_string(img, 70, True)
        img_strs.append(s)
    
    console = Console()
    for s in img_strs:
        console.clear()
        console.print(s, justify='center', width=100)
        time.sleep(0.1)
    # print(s)
