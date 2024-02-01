import os

import numpy as np
import pandas as pd
import scipy
from matplotlib.ticker import AutoMinorLocator
from scipy.ndimage import gaussian_filter
from scipy.interpolate import NearestNDInterpolator

CONST_PR_LABEL = 'PR'
CONST_PK_LABEL = 'PK'
CONST_X_LABEL = 'X'
CONST_Y_LABEL = 'Y'
CONST_Z_LABEL = 'Z'
CONST_S_LABEL = 'S'
CONST_SLOG_LABEL = 'S_log'
CONST_H_LABEL = 'H'
CONST_DIST_LABEL = 'DIST'
CONST_ALT_LABEL = 'ALT'
CONST_S_PR_LABEL = 'S_PR'
CONST_S_SUM_LABEL = 'S_SUM'


class STauProfile:
    """
    Класс для обработки данных STau по профилям
    """

    def __init__(self, path_to_files: ()):
        self.mask_below = None
        self._path_to_files = path_to_files

    def load_data(self, cut_to_depth: float = 0):
        """
        Прочитать и подготовить данные
        """
        self.__read_file_s_tau()
        self.__prepare_data(cut_to_depth)

    def get_path(self):
        return self._path_to_files[0]

    def __read_file_s_tau(self):
        data_list = []
        for path in self._path_to_files:
            # PR PK X Y Z Stau Altitude S_tau_PR SumS
            # 0  1  2 3 4  5    6         7        8
            use_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            names_cols = [CONST_PR_LABEL, CONST_PK_LABEL, CONST_X_LABEL, CONST_Y_LABEL, CONST_Z_LABEL, CONST_S_LABEL,
                          CONST_ALT_LABEL, CONST_S_PR_LABEL, CONST_S_SUM_LABEL]
            data = pd.read_table(path, delimiter='[ \t]', header=None, usecols=use_cols, names=names_cols, engine='python')
            data_list.append(data)
        self.data = pd.concat(data_list)
        

    def __sort_data(self):
        self.data = self.data.sort_values(by=[CONST_PR_LABEL, CONST_PK_LABEL, CONST_H_LABEL])

    def __prepare_data(self, cut_to_depth: float = 0):
        """
        Подготовить данные.
        Вычислить абсолютные отметки, дистанцию по профилю.
        Обрезать данные до глубины @cut_to_depth.
        :param cut_to_depth: Обрезать данные до этой глубины. Если None, то не обрезать
        """
        data = self.data
        # глубина
        data[CONST_H_LABEL] = data[CONST_Z_LABEL] - data[CONST_ALT_LABEL]

        # лог S
        data[CONST_SLOG_LABEL] = np.log10(data[CONST_S_LABEL])
        # дистанция
        x_start = data[CONST_X_LABEL].values[0]
        y_start = data[CONST_Y_LABEL].values[0]
        data[CONST_DIST_LABEL] = np.sqrt(np.power(data[CONST_X_LABEL] - x_start, 2) +
                                         np.power(data[CONST_Y_LABEL] - y_start, 2))
        # обрезка
        if cut_to_depth > 0:
            self.data = data.query(CONST_H_LABEL + '<=' + str(cut_to_depth))
        self.__sort_data()

    def interpolation(self, use_altitude: bool = True, grid_size: () = (500, 500),
                      use_mask: bool = True, use_average_filter: bool = False) -> ():
        """
        Интерполировать данные для создания грида и построения разреза.
        :param use_mask: Удалить области без данных.
        :param use_altitude: Строить по абсолютным отметкам или по глубине.
        :param grid_size: Размер грида в виде кортежа (num_X, num_Y).
        :param use_average_filter: Сгладить данные осреднением.
        :return: (X, Y, Z) для построения изоповерхности.
        """

        x, y, z = self.data_points(use_altitude)
        z = np.log10(z)
        # Интерполяция
        x_grid = np.linspace(min(x), max(x), grid_size[0])
        y_grid = np.linspace(min(y), max(y), grid_size[1])
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        interp = NearestNDInterpolator(list(zip(x, y)), z)
        z_mesh = interp(x_mesh, y_mesh)
        # Осреднение данных
        if use_average_filter:
            z_mesh = scipy.ndimage.generic_filter(z_mesh, function=np.mean, size=(13, 13), mode='nearest')
        z_mesh = np.power(10, z_mesh)
        # Маска
        if use_mask:
            delta_tolerance_m = 0
            mask = np.zeros_like(z_mesh, dtype=bool)
            mask_below = np.invert(np.zeros_like(z_mesh, dtype=bool))

            # По абсолютным отметкам или по глубине
            if use_altitude:
                data_upper_boundary = (
                    self.data.groupby([CONST_DIST_LABEL, CONST_PK_LABEL], as_index=False)[CONST_ALT_LABEL].max())
                data_bottom_boundary = (
                    self.data.groupby([CONST_DIST_LABEL, CONST_PK_LABEL], as_index=False)[CONST_ALT_LABEL].min())
                y_mask_upper = data_upper_boundary[CONST_ALT_LABEL].values
                y_mask_bottom = data_bottom_boundary[CONST_ALT_LABEL].values
            else:
                data_upper_boundary = (
                    self.data.groupby([CONST_DIST_LABEL, CONST_PK_LABEL], as_index=False)[CONST_H_LABEL].min())
                data_bottom_boundary = (
                    self.data.groupby([CONST_DIST_LABEL, CONST_PK_LABEL], as_index=False)[CONST_H_LABEL].max())
                y_mask_upper = data_upper_boundary[CONST_H_LABEL].values
                y_mask_bottom = data_bottom_boundary[CONST_H_LABEL].values
            x_mask = data_upper_boundary[CONST_DIST_LABEL].values

            x_relief_interp = x_mesh[0]
            y_relief_interp_upper = np.interp(x_relief_interp, x_mask, y_mask_upper)
            y_relief_interp_bottom = np.interp(x_relief_interp, x_mask, y_mask_bottom)

            for i, xi in enumerate(x_mesh):
                for ii, xii in enumerate(xi):
                    if (y_mesh[i, ii] > y_relief_interp_upper[ii] + delta_tolerance_m
                            or y_mesh[i, ii] < y_relief_interp_bottom[ii] - delta_tolerance_m):
                        mask[i, ii] = True
                    if y_mesh[i, ii] < y_relief_interp_bottom[ii]:
                        mask_below[i, ii] = False

            z_mesh = np.ma.masked_array(z_mesh, mask=mask)
            self.mask_below = mask_below

        return x_mesh, y_mesh, z_mesh

    def data_points(self, use_altitude: bool = True):
        return self.__data_for_altitude() if use_altitude else self.__data_for_depth()

    def __data_for_altitude(self):
        return self.data[CONST_DIST_LABEL], self.data[CONST_ALT_LABEL], self.data[CONST_S_LABEL]

    def __data_for_depth(self):
        return self.data[CONST_DIST_LABEL], self.data[CONST_H_LABEL], self.data[CONST_S_LABEL]

    def relief_points(self, use_altitude: bool = True):
        data_relief = self.data.groupby([CONST_DIST_LABEL, CONST_PK_LABEL], as_index=False)[CONST_Z_LABEL].max()
        x_relief = data_relief[CONST_DIST_LABEL].values
        y_relief = data_relief[CONST_Z_LABEL].values if use_altitude else np.zeros(len(x_relief))
        pk_relief = data_relief[CONST_PK_LABEL].values
        #pk_relief = [x if i%4==0 else "" for i,x in enumerate(pk_relief)]
        return x_relief, y_relief, pk_relief

    def relief_points_with_type(self, use_altitude: bool = True):
        data_relief = self.data.groupby([CONST_DIST_LABEL, CONST_PK_LABEL, 'type'], as_index=False)[CONST_Z_LABEL].max()
        x_relief = data_relief[CONST_DIST_LABEL].values
        y_relief = data_relief[CONST_Z_LABEL].values if use_altitude else np.zeros(len(x_relief))
        pk_relief = data_relief[CONST_PK_LABEL].values
        # pk_relief = [x if i%4==0 else "" for i,x in enumerate(pk_relief)]
        return x_relief, y_relief, pk_relief, data_relief['type'].values

    def get_profile(self):
        data_profile = set(self.data[CONST_PR_LABEL].values)
        return list(data_profile)

    def save_to_dat(self, path_to_save: str):
        if not os.path.isdir(os.path.dirname(path_to_save)):
            os.mkdir(os.path.dirname(path_to_save))
        self.data.to_csv(path_to_save, sep='\t', index=False)


class STauMap(STauProfile):
    """
    Класс для обработки данных STau по глубинам
    """

    def __init__(self, path_to_files: ()):
        super().__init__(path_to_files)
        self.z_levels_list = None

    def data_points_by_depth(self):
        depth_levels = self.z_levels_list
        results = []
        for i in range(len(depth_levels) - 1):
            data_level = self.data.query(f'{CONST_H_LABEL}>={depth_levels[i]}&{CONST_H_LABEL}<{depth_levels[i+1]}')
            group_mean = data_level.groupby([CONST_X_LABEL, CONST_Y_LABEL])[CONST_S_LABEL].mean().reset_index()
            x = group_mean[CONST_X_LABEL].values
            y = group_mean[CONST_Y_LABEL].values
            z = group_mean[CONST_S_LABEL].values
            results.append((x, y, z))
        return results

    def min_max_s_tau_value(self, percent_cut: float = 0):
        """
        Получить минимальное и максимальное значение s_tau
        :param percent_cut:
        :return:
        """
        z_list = []
        z_list.extend(self.data[CONST_S_LABEL])

        z_list = [float(a) for a in z_list]
        z_list.sort()
        n_z_list = len(z_list)
        cut_percent = percent_cut
        begin_n = round(n_z_list * cut_percent)
        end_n = n_z_list - round(n_z_list * cut_percent)
        z_list = z_list[begin_n: end_n]
        min_z = min(z_list)
        max_z = max(z_list)
        print(f'Set min and max of S values. Min = {min_z}. Max = {max_z} ')
        return min_z, max_z

    def load_data(self, cut_to_depth: float = 0, step_h: float = 50):
        """
        :param step_h: Через сколько метров строить горизонтальные срезы
        :return:
        """
        super().load_data()
        z_levels_list = [0]
        z_max = self.data[CONST_H_LABEL].max()
        while z_levels_list[-1] < z_max:
            z_levels_list.append(z_levels_list[-1] + step_h)
        self.z_levels_list = z_levels_list

    def interpolation(self, grid_size: () = (500, 500), use_mask: bool = True, use_average_filter: bool = True,
                      **kwargs) -> []:
        """
        Интерполировать данные для создания грида и построения горизонтальных срезов по глубине
        :param **kwargs:
        :param grid_size: Размер грида в виде кортежа (num_X, num_Y).
        :param use_mask: Удалить области без данных.
        :param use_average_filter: Сгладить данные осреднением.
        :return: list of ([X], [Y], [Z], h) для построения изоповерхностей.
        """

        xyz_levels = self.data_points_by_depth()
        res = []
        for xyz, h in zip(xyz_levels, self.z_levels_list):
            x, y, z = xyz
            z = np.log10(z)
            # Интерполяция
            x_grid = np.linspace(min(x), max(x), grid_size[0])
            y_grid = np.linspace(min(y), max(y), grid_size[1])
            x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
            interp = NearestNDInterpolator(list(zip(x, y)), z)
            z_mesh = interp(x_mesh, y_mesh)
            # Осреднение данных
            if use_average_filter:
                z_mesh = scipy.ndimage.generic_filter(z_mesh, function=np.mean, size=(13, 13), mode='nearest')
            z_mesh = np.power(10, z_mesh)
            res.append((x_mesh, y_mesh, z_mesh, h))
        return res




