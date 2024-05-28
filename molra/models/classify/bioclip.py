import torch
import open_clip

class BioCLIP:
    def __init__(self):
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')

    def run(self, image, text_labels):
        image = self.preprocess_val(image).unsqueeze(0) 
        text = self.tokenizer(text_labels)

        # Perform inference
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # max_prob, max_index = torch.max(text_probs, dim=1)
        # class_name = text_labels[max_index.item()].split(',')[0]
        # return class_name
        top_k = min(3, len(text_labels))  # Ensure we do not request more labels than available
        top_probs, top_indices = torch.topk(text_probs, top_k, dim=1)
        top_indices = top_indices.tolist()
        top_class_names = [text_labels[idx].split(',')[0] for idx in top_indices[0]]
        top_prob = top_probs[0][0].item()
        if top_prob > 0.4:
            return top_class_names
        else:
            return []

