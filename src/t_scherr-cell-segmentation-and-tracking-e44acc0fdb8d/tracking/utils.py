import os


def collect_paths(path_to_dir):
    """Returns a list of full paths to the lowest subdirectories of the provided path"""
    folder_content = os.walk(path_to_dir)
    sub_paths = [sub_path[0] for sub_path in folder_content if not sub_path[1]]
    for index in range(len(sub_paths)):
        sub_paths[index] = sub_paths[index].replace('\\', '/')

    return sub_paths
