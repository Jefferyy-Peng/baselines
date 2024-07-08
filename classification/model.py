from torch import nn
import torch


class ImageEncoderWithClass(nn.Module):
    def __init__(self, image_encoder):
        super(ImageEncoderWithClass, self).__init__()
        self.image_encoder = image_encoder
        self.embedding_size = image_encoder.pos_embed.shape[-1]
        self.head = ClassificationHead(self.embedding_size, 2)
        # freeze image encoder
        for param in self.image_encoder.parameters():
            param.require_grad = False
    
    def forward(self, input):
        embedding = self.image_encoder(input)
        output = self.head(embedding)
        return output

        

class ClassificationHead(nn.Module):
    def __init__(self, embedding_size, num_class):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(256, num_class)

    def forward(self, embedding):
        embedding = torch.mean(embedding.reshape(embedding.shape[0],embedding.shape[1], -1), dim=-1)
        output = self.fc(embedding)
        return output
