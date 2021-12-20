import os
import json

if __name__ == '__main__':
    dataset_name = '/raid5/lzl_data/someting-v2/something-something-v2'  # 'jester-v1'
    with open('%s-labels.json' % dataset_name) as f:
        data = json.load(f)
    categories = []
    for i, (cat, idx) in enumerate(data.items()):
        assert i == int(idx)  # make sure the rank is right
        categories.append(cat)

    with open('category.txt', 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    files_input = ['%s-validation.json' % dataset_name, '%s-train.json' % dataset_name, '%s-test.json' % dataset_name]
    files_output = ['val_videofolderv2.txt', 'train_videofolderv2.txt', 'test_videofolderv2.txt']
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            data = json.load(f)
        folders = []
        idx_categories = []
        for item in data:
            folders.append(item['id'])
            if 'test' not in filename_input:
                idx_categories.append(dict_categories[item['template'].replace('[', '').replace(']', '')])
            else:
                idx_categories.append(0)
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            dir_files = os.listdir(os.path.join('20bn-something-something-v2-frames', curFolder))
            output.append('%s %d %d' % (curFolder, len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))