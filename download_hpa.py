import os
import errno
from multiprocessing.pool import Pool
from tqdm import tqdm
import requests
from PIL import Image


def download(pid, image_list, base_url, save_dir, image_size=(512, 512)):
    colors = ['red', 'green', 'blue', 'yellow']

    for i in tqdm(image_list, postfix=pid):
        img_id = i.split('_', 1)
        for ind, color in enumerate(colors):
            try:
                img_path = img_id[0] + '/' + img_id[1] + '_' + color + '.jpg'
                img_name = i + '_' + color + '.png'
                img_url = base_url + img_path

                # Get the raw response from the url
                r = requests.get(img_url, allow_redirects=True, stream=True)
                r.raw.decode_content = True

                # Use PIL to resize the image and to convert it to L
                # (8-bit pixels, black and white)
                im = Image.open(r.raw)
                if color == 'yellow':
                    ind = 0
                im = im.resize(image_size, Image.LANCZOS).split()[ind]
                im.save(os.path.join(save_dir, img_name), 'PNG')
            except:
                print(i)
                print(img_url)


if __name__ == '__main__':
    # Parameters
    process_num = 1
    image_size  = (512, 512)
    url         = 'http://v18.proteinatlas.org/images/'
    # csv_path    = '../../../../kaggle_protein_atlas/input_data/hpa_website_set_2.csv'
    save_dir    = '../../../../kaggle_protein_atlas/input_data/train_extended_2/'

    # Create the directory to save the images in case it doesn't exist
    try:
        os.makedirs(save_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    print('Parent process %s.' % os.getpid())

    # img_list = pd.read_csv(csv_path)['Id']

    # external_raw = pd.read_csv('../../../../kaggle_protein_atlas/input_data/hpa_website_set_2.csv')
    # external_forum = pd.read_csv('../../../../kaggle_protein_atlas/input_data/HPAv18RBGY_wodpl.csv')
    # img_list = external_forum[~external_forum.Id.isin(external_raw.Id)].Id.values

    f = open("../../../../kaggle_protein_atlas/input_data/add_img_2.txt", 'r', encoding="utf8")
    m = f.readlines()
    f.close()

    # r = [(len(l.split('http://v18.proteinatlas.org/images/')), l) for l in m if 'http:' in l]

    img_list = m
    # for rr in r:
    #     if rr[0] == 2:
    #         img_list.append(
    #             "_".join(rr[1].split('http://v18.proteinatlas.org/images/')[-1].split('_')[:3]).replace('/', '_'))
    #     if rr[0] > 2:
    #         for i in rr[1].split('http://v18.proteinatlas.org/images/')[1:]:
    #             img_list.append("_".join(i.replace('/', '_').split('_')[:3]))

    list_len = len(img_list)
    p = Pool(process_num)

    for i in range(process_num):
        start = int(i * list_len / process_num)
        end = int((i + 1) * list_len / process_num)
        process_images = img_list[start:end]
        p.apply_async(download, args=(str(i), process_images, url, save_dir, image_size))

    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')