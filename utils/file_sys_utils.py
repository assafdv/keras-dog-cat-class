import glob
import os


def get_nb_files(directory, suffix):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*.{}".format(suffix))))
    return cnt


def get_files_recur(directory, suffix):
    """
    Get files by searching directory recursively
    """
    files = glob.glob(directory + '/**/*.{}'.format(suffix), recursive=True)
    return files