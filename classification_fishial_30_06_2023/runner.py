#Important
#the model was trained to classify cut out images of fish with a black background, so for the best possible classification I use only such images


import os
import cv2

from inference_class import EmbeddingClassifier

classification_package = ''
classification_path = os.path.join(classification_package, 'model.ckpt')
data_base_path = os.path.join(classification_package, 'embeddings.pt')
data_idx_path = os.path.join(classification_package, 'idx.json')
labels_path = os.path.join(classification_package, 'labels.json')

device = 'cpu'
model_classifier = EmbeddingClassifier(
        classification_path,
        data_base_path,
        labels_path,
        data_idx_path,
        device=device)

#your/image/path.jpeg
image_path = 'test_image.png'
#read image which you want to classify
im = cv2.imread(image_path)

#single image inference
single_output = model_classifier.inference_numpy(im) 

# you will get list of [['Tinca tinca', 1.0, ['008f26d7-ea82-5340-8902-6b0b20681892', '284620', 593428]]] where ['species_name',accuracy_between_0.0_and_1.0, Unnecessary info]

#batch image inference, remember about memmory if you have too big batch.
batch_output = model_classifier.batch_inference([im.copy(),im.copy(),im.copy(),im.copy(),im.copy()])

print(f"Single inference output: {single_output}")
print(f"Batch inference output: {batch_output}")
