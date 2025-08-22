import gzip, shutil, pathlib
src = pathlib.Path("C:/Users/Ananya/Downloads/HCAI-Project-main/HCAI-Project-main/project2/data/imdb.csv")
dst = src.with_suffix(src.suffix + ".gz")
with src.open("rb") as f_in, gzip.open(dst, "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)
print("Wrote:", dst, "Size:", dst.stat().st_size)
