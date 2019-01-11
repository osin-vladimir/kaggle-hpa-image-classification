import xml.etree.ElementTree as etree
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

PROTEINATLAS_XML_PATH = "../../../../kaggle_protein_atlas/input_data/proteinatlas.xml"
TRAIN_EXTRA_PATH      = "../../../../kaggle_protein_atlas/input_data/hpa_website_set.csv"

counter      = 0
data         = []
other_labels = defaultdict(int)

name_to_label_dict = {
                      'nucleoplasm'                   : 0,
                      'nuclear membrane'              : 1,
                      'nucleoli'                      : 2,
                      'nucleoli fibrillar center'     : 3,
                      'nuclear speckles'              : 4,
                      'nuclear bodies'                : 5,
                      'endoplasmic reticulum'         : 6,
                      'golgi apparatus'               : 7,
                      'peroxisomes'                   : 8,
                      'endosomes'                     : 9,
                      'lysosomes'                     : 10,
                      'intermediate filaments'        : 11,
                      'actin filaments'               : 12,
                      'focal adhesion sites'          : 13,
                      'microtubules'                  : 14,
                      'microtubule ends'              : 15,
                      'cytokinetic bridge'            : 16,
                      'mitotic spindle'               : 17,
                      'microtubule organizing center' : 18,
                      'centrosome'                    : 19,
                      'lipid droplets'                : 20,
                      'plasma membrane'               : 21,
                      'cell junctions'                : 22,
                      'mitochondria'                  : 23,
                      'aggresome'                     : 24,
                      'cytosol'                       : 25,
                      'cytoplasmic bodies'            : 26,
                      'rods & rings'                  : 27,
                      'vesicles'                      : 28,
                      'nucleus'                       : 29,
                      'midbody'                       : 30,
                      'midbody ring'                  : 31,
                      'cleavage furrow'               : 32
                     }

# Iterate over the XML file (since parsing it in one run might blow up the memory)
for event, elem in tqdm(etree.iterparse(PROTEINATLAS_XML_PATH, events=('start', 'end', 'start-ns', 'end-ns'))):
    if event == 'start':
        if elem.tag == "data" and len({"location", "assayImage"} - set([c.tag for c in elem.getchildren()])) == 0:
            labels = []
            assay_image = None
            for c in elem.getchildren():
                if c.tag == 'assayImage':
                    assay_image = c
                if c.tag == 'location':
                    if c.text in name_to_label_dict:
                        label = name_to_label_dict[c.text]
                        if type(label) is int:
                            labels.append(label)
                        else:
                            for l in label:
                                labels.append(l)
                    else:
                        other_labels[c.text] += 1

            if not labels:
                # Let's ignore images that do not have labels
                continue

            for image in assay_image.getchildren():
                if len(image.getchildren()) < 4 or image.getchildren()[-1].text is None:
                    continue
                image_url = image.getchildren()[-1].text

                assert "blue_red_green" in image_url

                for channel, color, object_ in zip(image.getchildren()[:-1], ["blue", "red", "green"],
                                                   ["nucleus", "microtubules", "antibody"]):
                    assert channel.text == object_
                    assert channel.attrib["color"] == color

                # "https://v18.proteinatlas.org/images/4109/24_H11_1_blue_red_green_yellow.jpg" -> "4109/24_H11_1"
                data.append(["/".join(image_url.split("/")[-2:]).replace("_blue_red_green.jpg", ""),
                             " ".join(str(x) for x in sorted(labels, reverse=True))])
                counter += 1

        # This is necessary to free up memory
        elem.clear()


print(counter)
print(other_labels)

df = pd.DataFrame(data=data, columns=["Id", "Target"])
df.to_csv(TRAIN_EXTRA_PATH, index=False)