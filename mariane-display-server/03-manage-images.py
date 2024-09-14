import os
import time
from PIL import Image

# TODO: adjust these
DIRECTORY = os.path.expanduser("~/Downloads")
FILETYPE = ".png"  # case-sensitive
PYPORTAL_PATH = "/Volumes/PYPORTAL"
MAX_IMAGES = 12  # max number of images to keep on device
TIME_BETWEEN_CHECKS = 10  # check every N seconds for new images

print("Starting up watching directory ", DIRECTORY)


def img2bmp(img_path):
    # weird mac bug
    # if "\u202f" in img_path:
    #     img_path = img_path.replace("\u202f", "")

    img = Image.open(img_path)
    if len(img.split()) == 4:
        # prevent IOError: cannot write mode RGBA as BMP
        r, g, b, a = img.split()
        img = Image.merge("RGB", (r, g, b))

    # image is now in memory as a BMP file
    # have to resize
    # base_width = 320
    # wpercent = (base_width / float(img.size[0]))
    # hsize = int((float(img.size[1]) * float(wpercent)))
    # img = img.resize((base_width, hsize), Image.Resampling.LANCZOS)

    # we're assuming fixed size
    img = img.resize((320, 180), Image.Resampling.LANCZOS)
    img = img.convert("P", palette=Image.ADAPTIVE, colors=255)

    return img


def get_files():
    return [x for x in os.listdir(os.path.join(PYPORTAL_PATH, "images")) if ".bmp" in x and not x.startswith(".")]


def get_latest_file():
    files = get_files()
    if len(files) == 0:
        return 0
    files.sort()
    idx = files[-1].split(".")[0][-4:]
    return int(idx)


def get_file_count():
    return len(get_files())


files = []
while True:
    # check for new files
    files_new = [x for x in os.listdir(DIRECTORY) if FILETYPE in x and not x.startswith(".")]
    if len(files_new) != len(files):
        diff = list(set(files_new) - set(files))  # assumes there's only ever more images
        print("new files", diff)

        for new_img in diff:
            print("handling new image", new_img)
            try:
                img = img2bmp(os.path.join(DIRECTORY, new_img))
            except Exception as e:  
                print(f"Error converting image {new_img}", e)
                continue

            # check latest filename on device
            last_idx = get_latest_file()

            # check file count on device
            no_files = get_file_count()

            print(f"found {no_files} images on device, last index is {last_idx}")
            # delete old file # assuming continuous file number
            if no_files == MAX_IMAGES:
                print(f"too many images on device, removing 'img{last_idx-MAX_IMAGES+1:04}.bmp'")
                os.remove(os.path.join(PYPORTAL_PATH, "images", f"img{last_idx-MAX_IMAGES+1:04}.bmp"))

            # send new file
            print(f"saving new image 'img{last_idx+1:04}.bmp'")
            img.save(os.path.join(PYPORTAL_PATH, "images", f"img{last_idx+1:04}.bmp"), format="BMP")
            files.append(new_img)

            print(f"...sleeping for {TIME_BETWEEN_CHECKS} seconds")
            time.sleep(TIME_BETWEEN_CHECKS)
