import os
import shutil
os.chdir('../data/')
path = os.getcwd()


def add_txt_to_clusters():
    dirs = os.listdir(path + '/clusters')
    for dir in dirs:
        files = os.listdir(path + f'/clusters/{dir}')
        for file in files:
            shutil.copy(path + f"/captcha_dataset_all/{file.replace('.png', '.txt', 1)}", path + f'/clusters/{dir}')


def upper_all_txt():
    dirs = os.listdir(path + '/clusters')
    for dir in dirs:
        files = os.listdir(path + f'/clusters/{dir}')
        for file in files:
            if '.txt' in file:
                text = ''
                with open(path + f'/clusters/{dir}/' + file, 'r+') as f:
                    text = f.read()[8:]
                with open(path + f'/clusters/{dir}/' + file, 'w') as f:
                    f.write(text)

def rename_request(hint, from_, to_):
    for i, j in enumerate(os.listdir(os.getcwd())):
        if hint in j:
            try:
                os.rename(j, j.replace(from_, to_, 1))
            except Exception:
                pass


def print_all():
    print(os.listdir(os.getcwd())[:-1])


def make_number():
    dir = os.listdir(os.getcwd())
    for i in range(len(dir) // 2):
        os.rename(dir[2 * i], str(i) + '.png')
        os.rename(dir[2 * i + 1], str(i) + '.txt')


def make_one_txt():
    res = []
    dir = os.listdir(os.getcwd())
    for i in range(len(dir) // 2):
        text_file = open(f'{i}.txt', "r")
        res += [text_file.read().upper()]

    write_file = text_file = open("real_train.txt", "w")
    write_file.write('#'.join(res))


dir = os.listdir(os.getcwd())


def rename_for_py_torch():
    for i, j in enumerate(dir):
        if '.png' in j:
            k = j.replace('_', '_request_', 1).replace('.png', '.txt', 1)
            os.rename(k, j.replace('.png', '.txt', 1))


def analyse_dataset(mode='check'):
    p = 0
    # print(dir)
    for i, j in enumerate(dir):
        k = ""
        if '.txt' in j:
            k = j.replace('_request_', '_', 1).replace('.txt', '.png', 1)
        if '.png' in j:
            k = j.replace('_', '_request_', 1).replace('.png', '.txt', 1)
        if k not in dir:
            if mode == 'check':
                print(j)
            elif mode == 'remove_single':
                os.remove(j)
            p += 1
    print(p)


def get_useless_txt():
    p = 0
    useless_files = []
    for i, j in enumerate(dir):
        if '.txt' in j:
            text_file = open(j, "r")
            if len(text_file.read()) != 4:
                # print(resp)
                p += 1
                useless_files += [j]
    print(p)
    return useless_files


def remove_useless(useless_files):
    for file in useless_files:
        try:
            os.remove(file)
            os.remove(file.replace('_request_', '_', 1).replace('.txt', '.png', 1))
        except Exception:
            pass


# analyse_dataset(mode='check')
# useless_files = get_useless_txt()
# remove_useless(useless_files)
# rename_for_py_torch()

# rename_request()
# remove_single()
# print_all()
# make_number()

#add_txt_to_clusters()
upper_all_txt()

# make_one_txt()

# rename_for_py_torch()
