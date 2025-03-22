import torch
import torch.nn as nn
import torch.nn.functional as models
from config import Config

class Encoder(nn.Module):
    def __init__(self, encoded_image_size=7):
        super(Encoder, self).__init__()


        # Load pretrained CNN
        if Config.CNN_ENCODER == "densenet121":
            cnn = models.densenet121(pretrained=True)
            self.enc_idm = 1024
            # Remove linear and pool layers
            modules = list(cnn.children())[:-1]
            self.cnn = nn.Sequential(*modules)

        elif Config.CNN_ENCODER == "resnet50":
            cnn = models.resnet50(pretrained=True)
            self.enc_idm = 2048

            # Remove linear and pool layers
            modules = list(cnn.children())[:-2]
            self.cnn = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unsupported CNN encoder: {Config.CNN_ENCODER}")
        
        # Resize output to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))


        # ?
        self.fine_tune()


    def forward(self, images):
        """
        Forward propogation
        images: [batch_size, 3, height, width]
        """
        features = self.cnn(images) # [batch_size, enc_dim, feat_h, feat_w]
        features = self.adaptive_pool(features) # [batch_size, enc_dim, enc_img_size]
        features = features.permute(0,2,3,1) # [batch_size, enc_img_size, enc_img_size, enc_dim]

        # Flatten spatial dimensions
        batch_size = features.size(0)
        features = features.view(batch_size, -1, self.enc_idm) # [batch_size, num_pixels, enc_dim]

        return features
    
    def fine_tune(self, fine_tune=True):
        """
        Fine-tune the CNN encoder
        """
        for param in self.cnn.parameters():
            param.requires_grad = fine_tune


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation
        encoder_out: [batch_size, num_pixels, encoder_dim]
        """
        att1 = self.encoder_att(encoder_out) # [batch_size, num_pixels, attention_dim]
        att2 = self.decoder_att(decoder_hidden)
        att2 = att2.unsqueeze(1)
        att = self.relu(att1+att2)
        att = self.full_att(att).squeeze(2)
        alpha = self.softmax(att)

        weighted_encoder_out = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return weighted_encoder_out, alpha
    
class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, encoder_dim, embed_dim = 512, decoder_dim =512, attention_dim=512, dropout=0.5):
        super(DecoderWithAttention, self).__init__()

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        # Attention mechanism
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # Decoder LSTM
        self.decode_step = nn.LSTMCell(embed_dim+encoder_dim, decoder_dim, bias=True)

        # Linear layers to find scores over vocabulary
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.fc  = nn.Linear(decoder_dim, vocab_size)
        self.sigmoid = nn.sigmoid()

        # Initialize weights
        self.init_weights()


    def init_weights(self):
        """
        Initialize weights for the model
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Initialize hidden state for encoder
        encoder__out: [batch_size, num_pixels, encoder_dim]
        """
        mean_encoder_out = encoder_out.mean(dim=1) # [batch_size, encoder_dim]
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation
        encoder_out: [batch_size, num_pizels, encoder_dim]
        encoded_captions: [batch_size, max_caption_length]
        captions_lengths: [batch_size, 1]
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        
        # Sort the input dta by decreasing caption length
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state
        h, c = self.init_hiddene_state(encoder_out)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths-1).tolist()

        # Create tensoes to hold word prediction scores
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(Config.DEVICE)
        alphas = torch.zeros(batch_size, max(encoded_captions), num_pixels, num_pixels).to(Config.DEVICE)


        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum((l > t for l in decode_lengths))

            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )

            gate = self.sigmoid(self.f_beta(h[:batch_size_t])) # gating scalar [batch_size_t, encoder_dim]


            attention_weighted_encoding = gate * attention_weighted_encoding
            
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )

            preds = self.fc(self.dropout(h)) # [batch_size_t, vocab_size]
            predictions[:batch_size_t, t, :]  = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions , encoded_captions, decode_lengths, alphas, sort_ind
    
    def predict(self, encoder_out, max_length=20):
        """
        Predict cpations for given images
        encoder_out: [1, num_pixels, encoder_dim]
        """
        batch_size = encoder_out.size(0)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)

        alphas = []
        predictions = []

        # Start token
        word = torch.tensor([1]).to(Config.DEVICE)
        emb = self.embedding(word) 

        for i in range(max_length):
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)

            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding

            h,c = self.decode_step(
                torch.cat([emb, attention_weighted_encoding], dim=1),
                (h,c)
            )

            preds = self.fc(h)
            predictions.append(preds)
            alphas.append(alpha)

            # Get next word with highest probability
            word = preds.argamx(dim=1)
            if word.item() == 2:
                break
                
            emb = self.embedding(word)

        return torch.cat([p.unsqueeze(1) for p in predictions], dim=1), torch.cat([a.unsqueeze for a in alphas], dim=1)
    

class MedicalCaptionModel(nn.Module):
    def __init__(self, vocab_size):
        super(MedicalCaptionModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = DecoderWithAttention(
            vocab_size=vocab_size,
            encoder_dim=self.encoder.enc_idm,
            embed_dim = Config.EMBEDDING_DIM,
            decoder_dim = Config.HIDDEN_DIM,
            attention_dim = Config.ATTENTION_DIM,
            dropout = Config.DROPOUT
        )

    def forward(self, images, captions, caption_lengths):
        """
        Forward propagation
        images: [batch_size, 3, height, width]
        captions: [batch_size, max_caption_length]
        caption_lengths: [batch_size, 1]
        """
        encoder_out = self.encoder(images)
        outputs = self.decoder(encoder_out, captions, caption_lengths)
        return outputs
    
    def predict(self, images, max_length=20):
        """
        Predict captions for given images
        images: [batch_size, 3, height, width]
        """
        encoder_out = self.encoder(images)
        return self.decoder.predict(encoder_out, max_length)






        

