import os
import sys

import s_tau
from paths_handler import PathsHandler
from s_tau import STauMap


def try_float(input_text: str) -> float | None:
    try:
        return float(input_text)
    except ValueError:
        return None


path_to_folder = ''
while not os.path.isdir(path_to_folder):
    path_to_folder = input('Введите путь до папки, где содержатся профили Stau: ')
    if not os.path.isdir(path_to_folder):
        print('Ошибка. Указанной папки не существует')
print(f'Указана папка: {path_to_folder}')

paths = PathsHandler.get_paths_all_merge(path_to_folder)
pr_len = len(PathsHandler.get_paths(path_to_folder, True))
answer = input(f'В указанной папке нашлось {pr_len} профилей. Продолжить? (Y/N): ')
while not (answer.upper() == 'Y' or answer.upper() == 'N'):
    answer = input(f'Введите Y или N:')

if answer.upper() == 'N':
    print('Завершение работы')
    sys.exit(1)

print('Объединение всех профилей...')
stau = STauMap([os.path.join(path_to_folder, x) for x in paths[0]])

step_h = input(f'Укажите шаг для осреднения, м: ')
while try_float(step_h) is None:
    step_h = input(f'Ошибка. Нужно ввести число. Укажите шаг для осреднения, м: ')

stau.load_data(step_h=float(step_h))
depth_levels = stau.z_levels_list
results = []
for i in range(len(depth_levels) - 1):
    data_level = stau.data.query(f'{s_tau.CONST_H_LABEL}>={depth_levels[i]}&{s_tau.CONST_H_LABEL}<{depth_levels[i+1]}')
    group = [s_tau.CONST_X_LABEL, s_tau.CONST_Y_LABEL, s_tau.CONST_PR_LABEL, s_tau.CONST_PK_LABEL]
    group_mean = data_level.groupby(group)[[s_tau.CONST_S_LABEL, s_tau.CONST_SLOG_LABEL]].mean().reset_index()
    group_mean = group_mean.sort_values(by=[s_tau.CONST_PR_LABEL, s_tau.CONST_PK_LABEL])
    if not os.path.isdir(os.path.join(path_to_folder, 'map')):
        os.mkdir(os.path.join(path_to_folder, 'map'))
    group_mean.to_csv(os.path.join(path_to_folder, 'map', f'map_{depth_levels[i]}.csv'), sep='\t', index=False)
    print('Срез сохранен: ' + os.path.join(path_to_folder, f'map_{depth_levels[i]}.csv'))
