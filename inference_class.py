import torch.nn as nn
import numpy as np
import logging
import torch
import json
import cv2

from PIL import Image
from torchvision import transforms
import torchvision.models as models


def read_json(path_to_json):
    with open(path_to_json) as f:
        return json.load(f)
        
class EmbeddingModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes = 185,  last_layer = 512, emb_dim=256):
        super().__init__()
        self.backbone = backbone
        self.embeddings = nn.Linear(last_layer, emb_dim)
        self.fc_parallel = nn.Linear(last_layer, num_classes)
        
    def forward(self, x: torch.Tensor):
        output_embedding = self.embeddings(self.backbone(x))
        output_fc = self.fc_parallel(self.backbone(x))
        return output_embedding, output_fc

class EmbeddingClassifier:
    def __init__(self, model_path, data_set_path, label_path, data_id_path, device='cpu', THRESHOLD = 8.84):
        self.device = device
        self.THRESHOLD = THRESHOLD
        self.model_path = model_path
        self.indexes_of_elements = read_json(data_id_path)
        self.labels_dict = read_json(label_path)
        self.softmax = nn.Softmax()
        
        resnet18 = models.resnet18()
        resnet18.fc = nn.Identity()

        self.model = EmbeddingModel(resnet18, num_classes = len(self.labels_dict),  last_layer = 512, emb_dim=256)
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device(device)), strict=False)
        self.model.eval()
        self.model.to(device)
        
        self.loader = transforms.Compose([
            transforms.Resize((224, 224), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.data_base = torch.load(data_set_path).to(device)
        logging.info("[INIT][CLASSIFICATION] Initialization of classifier was finished")
                
    def __inference(self, image, top_k = 15):
        logging.info("[PROCESSING][CLASSIFICATION] Getting embedding for a single detection mask")
        dump_embed, class_id = self.model(image.unsqueeze(0).to(self.device))
        class_id = self.softmax(class_id)
        logging.info("[PROCESSING][CLASSIFICATION] Classification by embedding for a single detection mask")
        output_by_embeddings = self.__classify(dump_embed.detach()[0])
        logging.info("[PROCESSING][CLASSIFICATION] Beautify output for a single detection mask")
        result = self.__beautifier_output(output_by_embeddings, class_id[0], top_k)
        return result
    
    def __beautifier_output(self, output_by_embeddings, class_id, top_k = 15):
        dict_results = {}
        for i in output_by_embeddings[:top_k]:
            if i[0] in dict_results:
                dict_results[i[0]]['values'].append(i[1].item())
                dict_results[i[0]]['annotations'].append(i[2])
            else:
                dict_results.update({i[0]: {'values': [i[1].item()],
                                           'annotations': [i[2]]}})

        for i in dict_results:
            dict_results[i].update({'top_1': dict_results[i]['values'][0]})
            dict_results[i].update({'annotation': dict_results[i]['annotations'][0]})
            dict_results[i].update({'median': np.median(dict_results[i]['values'])})
            del dict_results[i]['values']
            del dict_results[i]['annotations']
            
        dict_results = {self.labels_dict[str(label_id)]: dict_results[label_id] for label_id in dict_results}
        labels = self.__get_results(dict_results)
        classification_label = self.labels_dict[str(class_id.argmax().item())]
        labels = list(set(labels))
        for result in list(dict_results.keys()):
            if result not in labels:
                del dict_results[result]
            else:
                mean_distance = (dict_results[result]['top_1'] + dict_results[result]['median'])/2
                dict_results[result]['dist'] = mean_distance
                dict_results[result]['conf'] = round(self.__get_confidence(mean_distance), 3)
                logging.info(f"[PROCESSING][CLASSIFICATION] the threshold |{mean_distance}| has been recalculated to |{dict_results[result]['conf']}|")
        results = [[label, dict_results[label]['conf'], dict_results[label]['annotation']] for label in dict_results]
        logging.info("[PROCESSING][CLASSIFICATION] Classification by embedding was finished successful")
        if classification_label not in labels:
            logging.info("[PROCESSING][CLASSIFICATION] Append into output classification result by FC - layer")
            results.append([classification_label, 0.1, [None, None, None]])
        else:
            logging.info("[PROCESSING][CLASSIFICATION] Output from FC layer exist in Embedding results")
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results
    
    def __get_confidence(self, dist):
        min_dist = 4.2
        max_dist = self.THRESHOLD
        delta = max_dist - min_dist
        return 1.0 - (max(min(max_dist, dist), min_dist) - min_dist) / delta
    
    def inference_numpy(self, img, top_k=10):
        image = Image.fromarray(img)
        image = self.loader(image).float()
        image = torch.tensor(image)
        return self.__inference(image, top_k)
    
    def batch_inference(self, imgs):
        batch_input = []
        for idx in range(len(imgs)):  # assuming batch_size=len(imgs)
            image = Image.fromarray(imgs[idx])
            image = self.loader(image).float()
            image = torch.tensor(image)
            batch_input.append(image)

        batch_input = torch.stack(batch_input)
        dump_embeds, class_ids = self.model(batch_input)

        outputs = []
        for output_id in range(len(class_ids)):
            dump_embed, class_id = dump_embeds[output_id], class_ids[output_id]
            class_id = self.softmax(class_id)
            output_by_embeddings = self.__classify(dump_embed.detach())
            result = self.__beautifier_output(output_by_embeddings, class_id)
            outputs.append(result)
            
        return outputs
    
    def __classify(self, embedding):
        diff = (self.data_base - embedding).pow(2).sum(dim=2).sqrt()
        val, indi = torch.sort(diff)
        class_lib = []
        for idx, i in enumerate(val):
            for dist_id, dist in enumerate(i[:10]):
                ann_id = self.indexes_of_elements[str(idx)]['annotation_id'][indi[idx][dist_id]]
                draw_fish_id = self.indexes_of_elements[str(idx)]['drawn_fish_id'][indi[idx][dist_id]]
                img_id = self.indexes_of_elements[str(idx)]['image_id'][indi[idx][dist_id]]
                class_lib.append([idx, dist, [ann_id, img_id, draw_fish_id]])
                
        class_lib = sorted(class_lib, key=lambda x: x[1], reverse=False)
        return class_lib
    
    def __get_results(self, output):
        top_1, top_1_val = None, 10e9
        top_median, top_median_val = None, 10e9

        for i in output:
            if top_1_val > output[i]['top_1']:
                top_1 = i
                top_1_val = output[i]['top_1']

            if top_median_val > output[i]['median']:
                top_median = i
                top_median_val = output[i]['median']
        return [top_1, top_median]
