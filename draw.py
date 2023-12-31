import os
from time import sleep

from PIL import Image
from rich.console import Console
from rich.text import Text


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


class Draw:
    def __init__(self, console: Console, path: str):
        self.img_strs = []
        self.console = console
        self.path = path
        self.img2str()

    def img2str(self):
        self.img_strs = []
        for f in os.listdir(self.path):
            img = Image.open(os.path.join(self.path, f))
            s = to_string(img, 30, True)
            self.img_strs.append(Text.from_markup(s))

    def draw(self):
        for idx in range(len(self.img_strs))[::-1]:
            self.console.clear()
            self.console.print(self.img_strs[idx], justify='center', width=45, end='', no_wrap=True)
            sleep(0.05)
