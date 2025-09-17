import os.path
from datetime import datetime
from glob import glob

in_dir = "/home/tepex/PycharmProjects/energy/2022/source/dem"


def rename_grid(ext):
    files = glob(f"{in_dir}/2022*.{ext}")

    for old_file_name in sorted(files):
        bn = os.path.basename(old_file_name)
        # print(bn)
        s = bn.split("_")[0]

        dt = datetime.strptime(s, "%Y%m%d %H:%M:%S")
        new_s = dt.strftime("%Y%m%d %-H:%M:%S")
        # print(f"{s} -> {new_s}")

        new_file_name = os.path.join(in_dir, f"{new_s}_total.{ext}")
        print(new_file_name)

        os.rename(old_file_name, new_file_name)


if __name__ == "__main__":
    for ext in ("sgrd", "mgrd", "prj"):  # and the *.sdat
        rename_grid(ext)
