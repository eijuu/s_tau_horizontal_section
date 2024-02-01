import os


class PathsHandler:
    @staticmethod
    def get_paths(path_folder: str, merge_profile: bool = True) -> []:
        if os.path.isdir(path_folder):
            paths = PathsHandler.__get_paths_merge_profile(path_folder) if merge_profile \
                else PathsHandler.__get_paths_simple(path_folder)
            paths.sort(key=lambda a: int(a[0][:-4].split('_')[1]))
            return paths
        else:
            return []

    @staticmethod
    def __get_paths_merge_profile(path_folder: str) -> []:
        paths = []
        for root, dirs, files in os.walk(path_folder):
            for file in files:
                if file.endswith('.txt'):
                    flag = False
                    for i, element in enumerate(paths):
                        if file[:-4].split('_')[1] == element[0][:-4].split('_')[1]:
                            paths[i].extend([file])
                            flag = True
                    if not flag:
                        paths.append([file])
        return paths

    @staticmethod
    def __get_paths_simple(path_folder: str) -> []:
        paths = []
        for root, dirs, files in os.walk(path_folder):
            for file in files:
                if file.endswith('.txt'):
                    paths.append([file])
        return paths

    @staticmethod
    def get_paths_all_merge(path_folder: str):
        paths = []
        res = []
        for root, dirs, files in os.walk(path_folder):
            for file in files:
                if file.endswith('.txt'):
                    paths.extend([file])
        res.append(paths)
        return res