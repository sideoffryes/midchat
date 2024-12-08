from pypdf import PdfReader
import os
import re

# Reads all pdf files at data_path and returns a list of their file names
def getFiles(data_path: str) -> list:
    files = []

    for root, dirs, fnames in os.walk(data_path):
        for f in fnames:
            if f.rsplit('.', 1)[1] == "pdf":
                files.append(os.path.join(root, f))

    return files

def convertPdf(input_file: str):
    reader = PdfReader(input_file)
    line_pat = re.compile(r"\n{3,}")
    tab_pat = re.compile(r"\t{2,}")
    space_pat = re.compile(r" {2,}")
    
    parts = input_file.rsplit("/", 1)[1].split("_")
    inst_name = f"{parts[0]} {parts[1]}"
    
    text = []
    
    for page in reader.pages:
        text.append(page.extract_text(extraction_mode="layout", layout_mode_space_vertical=False))
    
    text = "".join(text)
    
    # get rid of non-ascii characters
    text = text.encode("ascii", "ignore").decode("ascii")
    
    # get ride of excessive new line characters and tabs
    text = line_pat.sub("\n\n", text)
    text = tab_pat.sub("\t", text)
    text = space_pat.sub(" ", text)
    
    # get rid of repeated inst name
    text = text.replace(inst_name, "")
    
    # get new path to save file
    fname = input_file.replace(".pdf", ".txt")
    fname = fname.replace("source", "text")
    
    # write new contents out to file
    with open(fname, 'w') as file:
        file.write(text)

# main method
def readSources() -> list:
    # default data path
    DATA_PATH = "./data/source"
    
    files = getFiles(DATA_PATH)
    new_names = []
    # convert each file to text
    for f in files:
        convertPdf(f)
        new = f.replace(".pdf", ".txt")
        new_names.append(new.replace("source", "text"))

    return new_names
    
if __name__ == "__main__":
    readSources()