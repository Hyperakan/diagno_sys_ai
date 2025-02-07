from unstructured.partition.auto import partition
import os
import re
from tqdm import tqdm

"""
Bu script verilen dokumanlardan metin kisimlarini cikartip txt dosyasina yazar.
Bu txt dosyalari vektor veri tabaninda indekslenecektir.
"""

def parse_document(file_path):
    elements = partition(filename=file_path)
    return "\n\n".join([str(el) for el in elements])


def clean_text(text):
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespaces
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    return text.strip()


def read_all_files_in_folder_extract_content(folder_path):
    contents = list()
    try:
        file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        for filename in tqdm(file_list, desc="Dosyalar okunuyor"):
            file_path = os.path.join(folder_path, filename)
            content = parse_document(file_path=file_path)
            if filename.endswith(".pdf"):
                filename = filename[:-4]
                txt = ".txt"
                txt_file_name = "".join((filename, txt))
                cleaned_content = clean_text(content)
            contents.append({"file_name": txt_file_name, "content": cleaned_content})
        return contents
    except Exception as e:
        print(f'Hata: {e}')
        return []
        


def write_files(contents: list, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    for item in tqdm(contents, desc="Dosyalar yazılıyor"):
        file_path = os.path.join(folder_path, item["file_name"])
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(item["content"])
            

documents = read_all_files_in_folder_extract_content("/home/onur-dev/Workspace/diagno_sys_ai/dummy_docs")

write_files(documents, "./extracted_txts")