# -*- coding: UTF-8 -*-
import os
import shutil

os.chdir('../data/')
path = os.getcwd()


def add_txt_to_clusters():
    dirs = os.listdir(path + '/clusters')
    for directory in dirs:
        files = os.listdir(path + f'/clusters/{directory}')
        for file in files:
            shutil.copy(path + f"/captcha_dataset_all/{file.replace('.png', '.txt', 1)}",
                        path + f'/clusters/{directory}')


def upper_all_txt():
    dirs = os.listdir(path + '/clusters')
    for directory in dirs:
        files = os.listdir(path + f'/clusters/{directory}')
        for file in files:
            if '.txt' in file:
                text = ''
                with open(path + f'/clusters/{directory}/' + file, 'r+') as read_file:
                    text = read_file.read()[8:]
                with open(path + f'/clusters/{directory}/' + file, 'w') as read_file:
                    read_file.write(text)


def rename_request(hint, from_, to_):
    for i, j in enumerate(os.listdir(os.getcwd())):
        if hint in j:
            try:
                os.rename(j, j.replace(from_, to_, 1))
            except OSError:
                pass
        if i % 5000:
            print(f'Renamed: {i}')


def rename(path):
    files = os.listdir(path)
    files = [os.path.join(path, x) for x in files if '.png' in x]

    os.mkdir(os.path.join(path, 'named'))
    for file in files:
        real_label = ''
        with open(file.replace('.png', '.txt'), 'r') as read_file:
            real_label = read_file.read()
        shutil.copy(file, os.path.join(path, 'named/' + real_label + '.png'))


def print_all():
    print(os.listdir(os.getcwd())[:-1])


def make_number():
    directory = os.listdir(os.getcwd())
    for i in range(len(directory) // 2):
        os.rename(directory[2 * i], str(i) + '.png')
        os.rename(directory[2 * i + 1], str(i) + '.txt')


def make_one_txt():
    res = []
    directory = os.listdir(os.getcwd())
    for i in range(len(directory) // 2):
        text_file = open(f'{i}.txt', "r")
        res += [text_file.read().upper()]

    write_file = text_file = open("real_train.txt", "w")
    write_file.write('#'.join(res))


directory = os.listdir(os.getcwd())


def analyse_dataset(mode='check'):
    count = 0
    # print(dir)
    for _, j in enumerate(directory):
        k = ""
        if '.txt' in j:
            k = j.replace('_request_', '_', 1).replace('.txt', '.png', 1)
        if '.png' in j:
            k = j.replace('_', '_request_', 1).replace('.png', '.txt', 1)
        if k not in directory:
            if mode == 'check':
                print(j)
            elif mode == 'remove_single':
                os.remove(j)
            count += 1
    print(count)


def get_useless_txt():
    count = 0
    useless_files = []
    for _, j in enumerate(directory):
        if '.txt' in j:
            text_file = open(j, "r")
            if len(text_file.read()) != 4:
                count += 1
                useless_files += [j]
    print(count)
    return useless_files


def remove_useless(useless_files):
    for file in useless_files:
        try:
            os.remove(file)
            os.remove(file.replace('_request_', '_', 1).replace('.txt', '.png', 1))
        except OSError:
            pass


if __name__ == '__main__':
    # analyse_dataset(mode='check')
    # useless_files = get_useless_txt()
    # remove_useless(useless_files)
    # rename_request()
    # remove_single()
    # print_all()
    # make_number()
    # add_txt_to_clusters()
    # upper_all_txt()
    # make_one_txt()
    pass
