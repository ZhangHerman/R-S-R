import os

if __name__ == '__main__':
    dataset_name = '/raid/ZYK-something-v1/something-v1/something-something-v1'  # 'jester-v1'
    with open('%s-labels.csv' % dataset_name) as f:
        lines = f.readlines()
    categories = []
    for line in lines:
        line = line.rstrip()
        categories.append(line)
    categories = sorted(categories)
    with open('category.txt', 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    files_input = ['%s-validation.csv' % dataset_name, '%s-train.csv' % dataset_name]
    files_output = ['val_videofolder.txt', 'train_videofolder.txt']
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            lines = f.readlines()
        folders = []
        idx_categories = []
        for line in lines:
            line = line.rstrip()
            items = line.split(';')
            folders.append(items[0])
            idx_categories.append(dict_categories[items[1]])
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            dir_files = os.listdir(os.path.join('/raid/ZYK-something-v1/something-v1/20bn-something-something-v1', curFolder))
            output.append('%s %d %d' % ('/raid/ZYK-something-v1/something-v1/20bn-something-something-v1/' + curFolder, len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))