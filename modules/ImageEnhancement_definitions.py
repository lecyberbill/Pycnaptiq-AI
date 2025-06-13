import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numbers
import numpy as np 
from einops import rearrange, repeat
import torchvision.transforms as transforms # Needed for Cross_Attention resize

# Import utility functions (assuming they are in Utils/utils.py relative to the main script)
# Or adjust the import based on your project structure
try:
     
    from Utils.utils import txt_color
except ImportError:
     
    def txt_color(text, _): return text
    print("[WARN] Could not import txt_color from Utils.utils in ImageEnhancement_models.py")

# --- Word Embedding Loading Functions (Moved from Helper) ---
def load_word_embeddings(file_path, vocab):
    """
    Loads word embeddings from a text file (like GloVe).
    Args:
    - file_path: Path to the embedding file.
    - vocab: List of words (type names) to load embeddings for.
    Returns:
    - Torch tensor of embeddings for the words in vocab.
    """
    embeddings = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in vocab:
                    vector = np.array(parts[1:], dtype=np.float32)
                    embeddings[word] = torch.from_numpy(vector)
    except FileNotFoundError:
        print(txt_color("[ERREUR]", "erreur"), f"Embedding file not found: {file_path}. Cannot initialize word embeddings.")
        raise FileNotFoundError(f"Required embedding file not found: {file_path}")

    embed_dim = 300 # Assuming GloVe 300d
    vocab_embeddings = torch.zeros(len(vocab), embed_dim)
    for i, word in enumerate(vocab):
        if word in embeddings:
            vocab_embeddings[i] = embeddings[word]
        else:
            print(txt_color("[AVERTISSEMENT]", "warning"), f"Word '{word}' not found in embedding file '{file_path}'. Using zero vector.")

    return vocab_embeddings

def initialize_wordembedding_matrix(wordembs_name, type_names):
    """
    Loads and combines word embeddings based on names.
    Args:
    - wordembs_name: Hyphen/plus separated word embedding names (e.g., 'glove'). '+' is used in original code example.
    - type_names: List of attributes/objects (vocab).
    """
    wordembs = wordembs_name.split('+')
    result = None
    total_dim = 0

    # Path relative to this models file, assuming structure modules/ImageEnhancement_models/embeddings/
    base_path = os.path.dirname(__file__)
    glove_file_path = os.path.join(base_path, "ImageEnhancement_models", "embeddings", "glove.6B.300d.txt")

    for wordemb_type in wordembs:
        wordemb_ = None
        current_dim = 0
        if wordemb_type == 'glove':
            print(f"[INFO] Loading GloVe embeddings from: {glove_file_path}")
            wordemb_ = load_word_embeddings(glove_file_path, type_names)
            current_dim = 300
        else:
             print(txt_color("[AVERTISSEMENT]", "warning"), f"Unknown word embedding type '{wordemb_type}' specified. Skipping.")

        if wordemb_ is not None:
            if result is None:
                result = wordemb_
            else:
                result = torch.cat((result, wordemb_), dim=1)
            total_dim += current_dim

    if result is None:
        print(txt_color("[ERREUR]", "erreur"), f"Failed to load any specified word embeddings: {wordembs_name}. Cannot initialize Embedder.")
        raise ValueError(f"Failed to load word embeddings: {wordembs_name}")

    print(f"[INFO] Final combined word embedding dimension: {total_dim}")
    return result, total_dim

# Import helper functions/classes defined within this file or standard libraries
# No external project-specific imports needed here if all classes are defined below.

# --- Helper Functions for Models ---
def to_3d(x):
	return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
	return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# --- LayerNorm Variants ---
class BiasFree_LayerNorm(nn.Module):
	def __init__(self, normalized_shape):
		super(BiasFree_LayerNorm, self).__init__()
		if isinstance(normalized_shape, numbers.Integral):
			normalized_shape = (normalized_shape,)
		normalized_shape = torch.Size(normalized_shape)
		assert len(normalized_shape) == 1
		self.weight = nn.Parameter(torch.ones(normalized_shape))
		self.normalized_shape = normalized_shape

	def forward(self, x):
		sigma = x.var(-1, keepdim=True, unbiased=False)
		return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
	def __init__(self, normalized_shape):
		super(WithBias_LayerNorm, self).__init__()
		if isinstance(normalized_shape, numbers.Integral):
			normalized_shape = (normalized_shape,)
		normalized_shape = torch.Size(normalized_shape)
		assert len(normalized_shape) == 1

		self.weight = nn.Parameter(torch.ones(normalized_shape))
		self.bias = nn.Parameter(torch.zeros(normalized_shape))
		self.normalized_shape = normalized_shape

	def forward(self, x):
		mu = x.mean(-1, keepdim=True)
		sigma = x.var(-1, keepdim=True, unbiased=False)
		return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
	def __init__(self, dim, LayerNorm_type):
		super(LayerNorm, self).__init__()
		if LayerNorm_type == 'BiasFree':
			self.body = BiasFree_LayerNorm(dim)
		else:
			self.body = WithBias_LayerNorm(dim)

	def forward(self, x):
		h, w = x.shape[-2:]
		return to_4d(self.body(to_3d(x)), h, w)

# --- Attention Mechanisms ---
class Cross_Attention(nn.Module):
    def __init__(self,
		 		dim,
				num_heads,
				bias,
				q_dim = 324): # Default query dimension from OneRestore code
        super(Cross_Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        # Ensure q_dim is a perfect square for resizing
        sqrt_q_dim_float = math.sqrt(q_dim)
        if sqrt_q_dim_float != int(sqrt_q_dim_float):
             raise ValueError(f"q_dim ({q_dim}) must be a perfect square for Cross_Attention resize.")
        sqrt_q_dim = int(sqrt_q_dim_float)

        self.resize = transforms.Resize([sqrt_q_dim, sqrt_q_dim], antialias=True) # Use antialias=True
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Linear(q_dim, q_dim, bias=bias)

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x, query):
        b,c,h,w = x.shape

        q = self.q(query)
        k, v = self.kv_dwconv(self.kv(x)).chunk(2, dim=1)
        k = self.resize(k)

        # Ensure q has the correct dimensions before repeat
        # Assuming query input is (b, q_dim)
        # q after self.q is (b, q_dim)
        # We need q to be compatible with k's shape after rearrange for matmul
        # k rearrange target: b head c (h w) -> (b, num_heads, dim//num_heads, sqrt_q_dim*sqrt_q_dim)
        # q needs to be: (b, num_heads, dim//num_heads, q_dim) ? No, matmul is (q @ k.transpose)
        # q needs to be: (b, num_heads, dim//num_heads, q_dim) -> This doesn't seem right.
        # Let's re-evaluate the original paper or code if possible.
        # Assuming the original einops `repeat(q, 'b l -> b head c l', ...)` intended q to be reshaped appropriately.
        # If q is (b, q_dim), maybe it should be projected first?
        # Sticking to the original code's likely intention for now, but this part might need review.
        # If q is (b, q_dim=324), repeat makes it (b, num_heads, dim//num_heads, 324)
        # k is (b, num_heads, dim//num_heads, 324)
        # v is (b, num_heads, dim//num_heads, 324)

        q = repeat(q, 'b l -> b head c l', head=self.num_heads, c=self.dim//self.num_heads) # This assumes q_dim == l
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        # The output shape depends on the matmul result. If q and k have last dim 324,
        # attn is (b, head, c, c). Then out is (b, head, c, 324).
        # Rearranging back needs the original h, w.
        # This seems inconsistent. Let's assume the rearrange target for 'out' should match 'x'.
        # Original code: rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # This implies the last dimension of 'out' must be h*w.
        # Let's trust the original rearrange and assume dimensions work out.
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class Self_Attention(nn.Module):
    def __init__(self,
		 		dim,
				num_heads,
				bias):
        super(Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

# --- FeedForward Network ---
class FeedForward(nn.Module):
	def __init__(self,
	      		dim,
				ffn_expansion_factor,
				bias):
		super(FeedForward, self).__init__()

		hidden_features = int(dim * ffn_expansion_factor)

		self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

		self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

		self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

	def forward(self, x):
		x = self.project_in(x)
		x1, x2 = self.dwconv(x).chunk(2, dim=1)
		x = F.gelu(x1) * x2
		x = self.project_out(x)
		return x

# --- Transformer Block ---
class TransformerBlock(nn.Module):
	def __init__(self,
	      		dim,
				num_heads=8,
				ffn_expansion_factor=2.66,
				bias=False,
				LayerNorm_type='WithBias'):
		super(TransformerBlock, self).__init__()
		self.norm1 = LayerNorm(dim, LayerNorm_type)
		self.cross_attn = Cross_Attention(dim, num_heads, bias)
		self.norm2 = LayerNorm(dim, LayerNorm_type)
		self.self_attn = Self_Attention(dim, num_heads, bias)
		self.norm3 = LayerNorm(dim, LayerNorm_type)
		self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

	def forward(self, x, query):
		x = x + self.cross_attn(self.norm1(x),query)
		x = x + self.self_attn(self.norm2(x))
		x = x + self.ffn(self.norm3(x))
		return x

# --- Residual Block ---
class ResidualBlock(nn.Module):
	def __init__(self, channel, norm=False): # norm parameter seems unused in original code
		super(ResidualBlock, self).__init__()
		self.el = TransformerBlock(channel, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')

	def forward(self, x,embedding):
		return self.el(x,embedding)

# --- OneRestore Components ---
class encoder(nn.Module):
	def __init__(self,channel):
		super(encoder,self).__init__()

		self.el = ResidualBlock(channel)#16
		self.em = ResidualBlock(channel*2)#32
		self.es = ResidualBlock(channel*4)#64
		self.ess = ResidualBlock(channel*8)#128

		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		self.conv_eltem = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)#16 32
		self.conv_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)#32 64
		self.conv_estess = nn.Conv2d(4*channel,8*channel,kernel_size=1,stride=1,padding=0,bias=False)#64 128
		# self.conv_esstesss = nn.Conv2d(8*channel,16*channel,kernel_size=1,stride=1,padding=0,bias=False)#128 256 # This seems unused

	def forward(self,x,embedding):

		elout = self.el(x, embedding)#16
		x_emin = self.conv_eltem(self.maxpool(elout))#32
		emout = self.em(x_emin, embedding)
		x_esin = self.conv_emtes(self.maxpool(emout))
		esout = self.es(x_esin, embedding)
		x_esin = self.conv_estess(self.maxpool(esout)) # Re-uses variable name x_esin, maybe intended?
		essout = self.ess(x_esin, embedding)#128

		return elout, emout, esout, essout

class backbone(nn.Module):
	def __init__(self,channel):
		super(backbone,self).__init__()

		self.s1 = ResidualBlock(channel*8)#128
		self.s2 = ResidualBlock(channel*8)#128

	def forward(self,x,embedding):

		share1 = self.s1(x, embedding)
		share2 = self.s2(share1, embedding)

		return share2

class decoder(nn.Module):
	def __init__(self,channel):
		super(decoder,self).__init__()

		self.dss = ResidualBlock(channel*8)#128
		self.ds = ResidualBlock(channel*4)#64
		self.dm = ResidualBlock(channel*2)#32
		self.dl = ResidualBlock(channel)#16

		self.conv_dsstds = nn.Conv2d(8*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)#128 64
		self.conv_dstdm = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)#64 32
		self.conv_dmtdl = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)#32 16

	def _upsample(self,x,y):
		_,_,H0,W0 = y.size()
		return F.interpolate(x,size=(H0,W0),mode='bilinear', align_corners=False) # Added align_corners=False

	def forward(self, x, x_ss, x_s, x_m, x_l, embedding):

		dssout = self.dss(x + x_ss, embedding)
		x_dsin = self.conv_dsstds(self._upsample(dssout, x_s))
		dsout = self.ds(x_dsin + x_s, embedding)
		x_dmin = self.conv_dstdm(self._upsample(dsout, x_m))
		dmout = self.dm(x_dmin + x_m, embedding)
		x_dlin = self.conv_dmtdl(self._upsample(dmout, x_l))
		dlout = self.dl(x_dlin + x_l, embedding)

		return dlout

# --- Main OneRestore Model ---
class OneRestore(nn.Module):
	def __init__(self, channel = 32):
		super(OneRestore,self).__init__()
		self.norm = lambda x: (x-0.5)/0.5
		self.denorm = lambda x: (x+1)/2
		self.in_conv = nn.Conv2d(3,channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.encoder = encoder(channel)
		self.middle = backbone(channel)
		self.decoder = decoder(channel)
		self.out_conv = nn.Conv2d(channel,3,kernel_size=1,stride=1,padding=0,bias=False)

	def forward(self,x,embedding):
		# x is expected to be in range [0, 1]
		x_in = self.in_conv(self.norm(x))
		x_l, x_m, x_s, x_ss = self.encoder(x_in, embedding)
		x_mid = self.middle(x_ss, embedding)
		x_out = self.decoder(x_mid, x_ss, x_s, x_m, x_l, embedding)
		# Residual connection added to the *original* input x before denormalization
		out = self.out_conv(x_out) + x
		# Denormalize the final output
		return self.denorm(out)

# --- Backbone for Embedder ---
class Backbone(nn.Module):
    def __init__(self, backbone='resnet18'):
        super(Backbone, self).__init__()

        # Use updated weights API
        if backbone == 'resnet18':
            resnet = torchvision.models.resnet.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            self.feat_dim = 512 # ResNet18 output feature dimension before avgpool
        elif backbone == 'resnet50':
            resnet = torchvision.models.resnet.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2) # Use correct weights
            self.feat_dim = 2048 # ResNet50 output feature dimension
        elif backbone == 'resnet101':
            resnet = torchvision.models.resnet.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2) # Use correct weights
            self.feat_dim = 2048 # ResNet101 output feature dimension
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")

        # Keep only the layers needed for feature extraction up to block4
        self.block0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        )
        self.block1 = resnet.layer1
        self.block2 = resnet.layer2
        self.block3 = resnet.layer3
        self.block4 = resnet.layer4

    def forward(self, x):
        # The Embedder expects the output of the last block (block4)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # Embedder expects a list/tuple, return the final feature map
        return [x]

# --- Cosine Classifier for Embedder ---
class CosineClassifier(nn.Module):
    def __init__(self, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.temp = temp

    def forward(self, img, concept, scale=True):
        """
        img: (bs, emb_dim)
        concept: (n_class, emb_dim)
        """
        img_norm = F.normalize(img, dim=-1)
        concept_norm = F.normalize(concept, dim=-1)
        pred = torch.matmul(img_norm, concept_norm.transpose(0, 1))
        if scale:
            pred = pred / self.temp
        return pred

# --- Embedder Model ---
class Embedder(nn.Module):
    """
    Text and Visual Embedding Model. (Integrated from OneRestore source)
    Depends on Backbone, initialize_wordembedding_matrix (defined above), CosineClassifier.
    """
    def __init__(self,
                 type_name, # Corresponds to combine_type in load_embedder_ckpt
                 feat_dim = 512, # This will be overridden by Backbone's feat_dim
                 mid_dim = 1024,
                 out_dim = 324,
                 drop_rate = 0.35,
                 cosine_cls_temp = 0.05,
                 wordembs = 'glove', # Default from provided code
                 extractor_name = 'resnet18'): # Default from provided code
        super(Embedder, self).__init__()

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.type_name = type_name
        # self.feat_dim will be set in _setup_image_embedding
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.drop_rate = drop_rate
        self.cosine_cls_temp = cosine_cls_temp
        self.wordembs = wordembs
        self.extractor_name = extractor_name
        # Use torchvision.transforms directly
        self.transform = transforms.Normalize(mean, std)

        # Word embedding setup uses initialize_wordembedding_matrix defined in this file
        self._setup_word_embedding(initialize_wordembedding_matrix)
        self._setup_image_embedding()

    def _setup_image_embedding(self):
        # image embedding
        self.feat_extractor = Backbone(backbone=self.extractor_name)
        self.feat_dim = self.feat_extractor.feat_dim # Get actual feature dim

        img_emb_modules = [
            nn.Conv2d(self.feat_dim, self.mid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.mid_dim),
            nn.ReLU()
        ]
        if self.drop_rate > 0:
            img_emb_modules += [nn.Dropout2d(self.drop_rate)]
        self.img_embedder = nn.Sequential(*img_emb_modules)

        self.img_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_final = nn.Linear(self.mid_dim, self.out_dim)

        self.classifier = CosineClassifier(temp=self.cosine_cls_temp)

    def _setup_word_embedding(self, init_word_emb_func):
        self.type2idx = {self.type_name[i]: i for i in range(len(self.type_name))}
        self.num_type = len(self.type_name)
        train_type = [self.type2idx[type_i] for type_i in self.type_name]
        self.register_buffer('train_type', torch.LongTensor(train_type))

        # Use the local function (passed as argument for consistency)
        wordemb, self.word_dim = init_word_emb_func(self.wordembs, self.type_name)

        self.embedder = nn.Embedding(self.num_type, self.word_dim)
        self.embedder.weight.data.copy_(wordemb)

        self.mlp = nn.Sequential(
                nn.Linear(self.word_dim, self.out_dim),
                nn.ReLU(True)
            )

    # --- Inference Methods ---
    def image_encoder_forward(self, batch):
        img = self.transform(batch.to(next(self.parameters()).device))
        scene_emb = self.embedder(self.train_type)
        scene_weight = self.mlp(scene_emb)
        img_features = self.feat_extractor(img)[0]
        bs, _, h, w = img_features.shape
        img_embedded = self.img_embedder(img_features)
        img_pooled = self.img_avg_pool(img_embedded).squeeze(3).squeeze(2)
        img_final_features = self.img_final(img_pooled)
        pred_logits = self.classifier(img_final_features, scene_weight)
        pred_indices = torch.max(pred_logits, dim=1)[1]
        out_embedding = scene_weight[pred_indices]
        num_type = self.train_type[pred_indices]
        text_type = [self.type_name[idx.item()] for idx in num_type]
        return out_embedding, num_type, text_type

    def text_encoder_forward(self, text_list):
        bs = len(text_list)
        device = next(self.parameters()).device
        scene_emb = self.embedder(self.train_type)
        scene_weight = self.mlp(scene_emb)
        num_type_list = [self.type2idx[text] for text in text_list]
        num_type = torch.tensor(num_type_list, dtype=torch.long, device=device)
        out_embedding = scene_weight[num_type]
        text_type = text_list
        return out_embedding, num_type, text_type

    # --- Other Methods (Simplified/Placeholder) ---
    def train_forward(self, batch):
        print("[WARN] train_forward called - not implemented for inference focus.")
        return {'loss_total': torch.tensor(0.0), 'acc_type': torch.tensor(0.0)}

    def text_idx_encoder_forward(self, idx):
        print("[WARN] text_idx_encoder_forward called - not implemented for inference focus.")
        bs = idx.shape[0]
        device = next(self.parameters()).device
        scene_emb = self.embedder(self.train_type)
        scene_weight = self.mlp(scene_emb)
        return scene_weight[idx.to(device)]

    def contrast_loss_forward(self, batch):
        print("[WARN] contrast_loss_forward called - not implemented for inference focus.")
        img = self.transform(batch.to(next(self.parameters()).device))
        img = self.feat_extractor(img)[0]
        img = self.img_embedder(img)
        img = self.img_avg_pool(img).squeeze(3).squeeze(2)
        img = self.img_final(img)
        return img

    def forward(self, x, type = 'image_encoder'):
        with torch.no_grad(): # Ensure no gradients for all inference paths
            if type == 'train':
                # Should not happen in inference, but handle gracefully
                out = self.train_forward(x)
            elif type == 'image_encoder':
                out = self.image_encoder_forward(x)
            elif type == 'text_encoder':
                out = self.text_encoder_forward(x)
            elif type == 'text_idx_encoder':
                out = self.text_idx_encoder_forward(x)
            elif type == 'visual_embed':
                x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                out = self.contrast_loss_forward(x_resized)
            else:
                raise ValueError(f"Unknown forward type: {type}")
        return out
