import argparse
import csv
import os
import json
import cv2

def get_filelist(file_path):
    Filelist = []
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonfile", type=str)
    args = parser.parse_args()
    f = open(args.jsonfile)
    data = json.load(f)

    csvpath =  os.path.join(os.path.dirname(args.jsonfile), "annotations.csv")
    csvf = open(csvpath, "w")
    writer = csv.writer(csvf)
    n = 0
    for videopair in data:
        video = videopair["file"]
        assert(len(videopair["captions"]) == 1)
        caption = videopair["captions"][0].replace(",", ".")  
        videop = os.path.join(os.path.dirname(args.jsonfile), "videos", video)
        cap = cv2.VideoCapture(videop)  
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        writer.writerow([videop, caption, frame_count])
        n = n + 1
        if(n % 100 == 0):
            print("processing " + str(n) + "\n")
    csvf.close()
