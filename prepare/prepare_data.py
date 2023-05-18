import random
import os
import shutil
import xml.etree.ElementTree as ET
import tqdm


def makedir(newdir):
    if not os.path.exists(newdir):
        os.makedirs(newdir)


def delete_cache(path):
    for filename in os.listdir(path):
        if filename.endswith(".cache"):
            os.remove(os.path.join(path, filename))


def onexml2yolo(xmlpath, classname):
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    size = root.find("size")
    width = int(float(size.find("width").text))
    height = int(float(size.find("height").text))
    newlines = []
    for oneobject in root.findall("object"):
        bndbox = oneobject.find("bndbox")
        name = oneobject.find("name").text
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))
        xcenter = (xmax + xmin) / (2. * width)
        ycenter = (ymax + ymin) / (2. * height)
        yolowidth = (xmax - xmin) / width
        yoloheight = (ymax - ymin) / height
        if name not in classname.keys():
            continue
        line = f"{classname[name]} " + f"{xcenter:.6f} {ycenter:.6f} {yolowidth:.6f} {yoloheight:.6f}"
        newlines.append(line)
    return newlines


def generate_yolo_label(xmlpath, outpath, classname, dataname):
    makedir(outpath)
    progress_bar = tqdm.tqdm(desc=f"{dataname} xml2yolo", total=len(os.listdir(xmlpath)))
    for xml in os.listdir(xmlpath):
        newname = xml.split(".")[0]
        newname = newname + ".txt"
        xml = os.path.join(xmlpath, xml)
        newlines = onexml2yolo(xml, classname)
        with open(os.path.join(outpath, newname), "w") as f:
            for line in newlines:
                f.write(line + "\n")
        progress_bar.update()
    progress_bar.close()


def split_dataset(train_ratio, images_dir):
    random.seed(42)
    imglist = os.listdir(images_dir)
    random.shuffle(imglist)
    trainset = []
    testset = []
    train_num = int(train_ratio*len(imglist))
    for i in range(len(imglist)):
        img = os.path.join(images_dir, imglist[i])
        if i + 1 <= train_num:
            trainset.append(img)
        else:
            testset.append(img)
    valset = trainset[-5:]
    return trainset, valset, testset


if __name__ == "__main__":
    rdd2020 = os.path.join("../datasets/RDD2020/train")
    sodrv1 = os.path.join("../datasets/SODRv1")
    rdddatasets = ["Czech", "India", "Japan"]
    sodrdatasets = ["collapse", "blockage"]
    classname = {"D00": 0, "D10": 1, "D20": 2, "D40": 3, "collapse": 4, "blockage": 5}   # note that D50: collapse D60: blockage

    # xml to yolo
    for d in rdddatasets:
        xmls = os.path.join(rdd2020, d, "annotations", "xmls")
        outputyolo = os.path.join(rdd2020, d, "labels")
        generate_yolo_label(xmls, outputyolo, classname, "RDD2020_" + d)
    for d in sodrdatasets:
        xmls = os.path.join(sodrv1, d, "xmls")
        outputyolo = os.path.join(sodrv1, d, "labels")
        generate_yolo_label(xmls, outputyolo, classname, "SODRv1_" + d)

    # split train, val, and test
    delete_cache("../datasets")
    train_ratio = 0.8
    test_ratio = 0.2
    train = []
    val = []
    test = []
    for d in rdddatasets:
        imgs = os.path.join(rdd2020, d, "images")
        trainset, valset, testset = split_dataset(train_ratio, imgs)
        train.extend(trainset)
        val.extend(valset)
        test.extend(testset)
    for d in sodrdatasets:
        imgs = os.path.join(sodrv1, d, "images")
        trainset, valset, testset = split_dataset(train_ratio, imgs)
        train.extend(trainset)
        val.extend(valset)
        test.extend(testset)
    with open("../datasets/train.txt", "w") as f:
        for i in train:
            f.write(i.lstrip("../")+"\n")
    with open("../datasets/val.txt", "w") as f:
        for i in val:
            f.write(i.lstrip("../")+"\n")
    with open("../datasets/test.txt", "w") as f:
        for i in test:
            f.write(i.lstrip("../")+"\n")


