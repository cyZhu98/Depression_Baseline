import torch
import torch.nn as nn
import transformers
import torchvision


class Text_Split(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = transformers.AutoModelForSequenceClassification.from_pretrained(
            'hfl/chinese-xlnet-base')
        # self.text_encoder = transformers.AutoModelForSequenceClassification.from_pretrained(
        #     'hfl/chinese-roberta-wwm-ext')
        self.text_encoder.logits_proj = nn.Linear(768, 128)
        # self.text_encoder.classifier = nn.Linear(768, 128)  # For RoBERTa
        self.gru = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, txt, image=None, attention_mask=None):
        '''
        input:
            x: text_ids [Batch, Count, len]
            image: ignored
            attention_mask : [Batch, Count, len]
        '''
        features = []
        for num in range(txt.shape[1] - 1, -1, -1):
            txt_in = txt[:, num]
            # features [Count, Batch, 768]
            features.append(self.text_encoder(
                txt_in, attention_mask=attention_mask[:, num]).logits)
        features = torch.stack(features).permute(1, 0, 2)
        x, hidden = self.gru(features)
        x = x[:, -1]
        x = self.fc(x)
        return x


class Text_Splice_Finetune(nn.Module):
    '''
    Fine-tune the XLNet-Base, you can change it to the RoBERTa
    input : 
        text : [batch, length]
    '''
    def __init__(self):
        super().__init__()
        self.text_encoder = transformers.AutoModelForSequenceClassification.from_pretrained(
            'hfl/chinese-xlnet-base')
        # self.text_encoder = transformers.AutoModelForSequenceClassification.from_pretrained(
        #     'hfl/chinese-roberta-wwm-ext')
    def forward(self, txt, image=None, attention_mask=None):
        x = self.text_encoder(txt, attention_mask=attention_mask)  # x [Batch, length, 768]
        return x.logits


class Text_Splice_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = transformers.AutoModelForSequenceClassification.from_pretrained(
            'hfl/chinese-xlnet-base').transformer.word_embedding
        self.gru = nn.GRU(768, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 2)

    def forward(self, txt, image=None, attention_mask=None):
        x = self.text_encoder(txt)  # x [Batch, length, 768]
        x, hidden = self.gru(x)
        x = x[:, -1]
        x = self.fc(x)
        return x.logits


# TODO:
class Image_Only(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.resnet50(pretrained=True)
        self.encoder.fc = nn.Sequential(nn.Linear(2048, 128),
                                        nn.Tanh())
        self.gru = nn.GRU(128, 50, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(100, 2)

    def forward(self, text, image, text_att=None):
        '''
        input:
            text: ignored
            image: [Batch, Count, 3, 224, 224]
        '''
        features = []
        for num in range(image.shape[1] - 1, -1, -1):  # Reverse
            img_in = image[:, num]  # [B, C, H, W]
            features.append(self.encoder(img_in))  # [B, C_out]
        features = torch.stack(features)
        x, hidden = self.gru(features)
        x = x[-1]
        x = self.fc(x)
        return x


class Fusion_Basic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_I = torchvision.models.resnet50(pretrained=True)
        self.encoder_I.fc = nn.Sequential(nn.Linear(2048, 128),
                                          nn.Tanh())

        self.encoder_T = transformers.AutoModelForSequenceClassification.from_pretrained(
            'hfl/chinese-roberta-wwm-ext')
        self.encoder_T.classifier = nn.Linear(768, 128)

        self.gru = nn.GRU(128, 50, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(100, 2)

    def forward(self, text, image, att_mask=None):
        '''
        input:
            text: ignored
            image: [Batch, Count, 3, 224, 224]
        '''
        features_I = []
        for num in range(image.shape[1] - 1, -1, -1):  # Reverse
            img_in = image[:, num]  # [B, C, H, W]
            features_I.append(self.encoder_I(img_in))  # [B, C_out]
        features_I = torch.stack(features_I)

        features_T = []
        for num in range(text.shape[1] - 1, -1, -1):
            txt_in = text[:, num]
            features_T.append(self.text_encoder(
                txt_in, attention_mask=att_mask[:, num]).logits)  # [Batch, Count, 768]
        features_T = torch.stack(features_T)

        x, hidden = self.gru(features)
        x = x[-1]
        x = self.fc(x)
        return x


if __name__ == '__main__':
    test_net = Text_Split()
    inputs = torch.randint(0, 100, (3, 256))
    print(test_net)
