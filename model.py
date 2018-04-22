import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init


#========================================Knowing When to Look========================================
class AttentiveCNN( nn.Module ):
    def __init__( self, embed_size, hidden_size ):
        super( AttentiveCNN, self ).__init__()
        
        # ResNet-152 backend
        resnet = models.resnet152( pretrained=True )
        modules = list( resnet.children() )[ :-2 ] # delete the last fc layer and avg pool.
        resnet_conv = nn.Sequential( *modules ) # last conv feature
        
        self.resnet_conv = resnet_conv
        self.avgpool = nn.AvgPool2d( 7 )
        self.affine_a = nn.Linear( 2048, hidden_size ) # v_i = W_a * A
        self.affine_b = nn.Linear( 2048, embed_size )  # v_g = W_b * a^g
        
        # Dropout before affine transformation
        self.dropout = nn.Dropout( 0.5 )
        
        self.init_weights()
        
    def init_weights( self ):
        """Initialize the weights."""
        init.kaiming_uniform( self.affine_a.weight, mode='fan_in' )
        init.kaiming_uniform( self.affine_b.weight, mode='fan_in' )
        self.affine_a.bias.data.fill_( 0 )
        self.affine_b.bias.data.fill_( 0 )
        
        
    def forward( self, images ):
        '''
        Input: images
        Output: V=[v_1, ..., v_n], v_g
        '''
        
        # Last conv layer feature map
        A = self.resnet_conv( images )
        
        # a^g, average pooling feature map
        a_g = self.avgpool( A )
        a_g = a_g.view( a_g.size(0), -1 )
        
        # V = [ v_1, v_2, ..., v_49 ]
        V = A.view( A.size( 0 ), A.size( 1 ), -1 ).transpose( 1,2 )
        V = F.relu( self.affine_a( self.dropout( V ) ) )
        
        v_g = F.relu( self.affine_b( self.dropout( a_g ) ) )
        
        return V, v_g

# Outfit Encoder
class OEncoder(nn.Module):
	def __init__(self, embed_size, hidden_size):
		super(OEncoder, self).__init__()

		# GRU Encoder:
		self.GRU = nn.GRU(embed_size, hidden_size)

	def forward(self, images, states=None):

		# Encoder given outfit
		_, encoder_output = self.GRU(images, states)

		return encoder_output
    

# Caption Decoder
class Decoder( nn.Module ):
    def __init__( self, embed_size, vocab_size, hidden_size ):
        super( Decoder, self ).__init__()

        # word embedding
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # GRU Decoder:
    	self.GRU = nn.GRU(embed_size, hidden_size)
        
        # Save hidden_size for hidden and cell variable 
        self.hidden_size = hidden_size

	# Linear transformation
	self.output = nn.Linear(hidden_size, vocab_size)        
        
    def forward(self, outfit, states=encoder_output):
        
        # Word Embedding
        embeddings = self.embed(outfit)
        
        # Decoder given outfit
	decoder_output, _ = self.GRU(embeddings, states)

	scores = self.output(decoder_output)
   
        # Return states for sampling purpose
        return scores, states
    
        

# Whole Architecture with Image Encoder and Caption decoder        
class Encoder2Decoder( nn.Module ):
    def __init__( self, embed_size, vocab_size, hidden_size ):
        super( Encoder2Decoder, self ).__init__()

	self.embed_size = embed_size
        
        # Image CNN encoder and Adaptive Attention Decoder
        self.image_encoder = AttentiveCNN(embed_size, hidden_size)
	self.outfit_encoder = OEncoder(embed_size, hidden_size)
        self.decoder = Decoder(embed_size, vocab_size, hidden_size)
        
        
    def forward(self, images, outfits, lengths):        
        # Data parallelism for V v_g encoder if multiple GPUs are available
        # V=[ v_1, ..., v_k ], v_g in the original paper
	iembeddings = Variable(torch.zeros(images.size(0), images.size(1), self.embed_size))
	for a in range(images.size(1)):
		bimages = images[:, a, :]
        	if torch.cuda.device_count() > 1:
            		device_ids = range( torch.cuda.device_count() )
            		encoder_parallel = torch.nn.DataParallel( self.encoder, device_ids=device_ids )
            		V, v_g = encoder_parallel(bimages) 
        	else:
            		V, v_g = self.encoder(bimages)
		iembeddings[:, a, :] = v_g
        
        # Outfit prediction
        scores, _, _,_ = self.decoder( V, v_g, captions )
        
        # Pack it to make criterion calculation more efficient
        packed_scores = pack_padded_sequence( scores, lengths, batch_first=True )
        
        return packed_scores
    
    # Caption generator
    def sampler( self, images, max_len=20 ):
        """
        Samples captions for given image features (Greedy search).
        """
        
        # Data parallelism if multiple GPUs
        if torch.cuda.device_count() > 1:
            device_ids = range( torch.cuda.device_count() )
            encoder_parallel = torch.nn.DataParallel( self.encoder, device_ids=device_ids )
            V, v_g = encoder_parallel( images ) 
        else:    
            V, v_g = self.encoder( images )
            
        # Build the starting token Variable <start> (index 1): B x 1
        if torch.cuda.is_available():
            captions = Variable( torch.LongTensor( images.size( 0 ), 1 ).fill_( 1 ).cuda() )
        else:
            captions = Variable( torch.LongTensor( images.size( 0 ), 1 ).fill_( 1 ) )
        
        # Get generated caption idx list, attention weights and sentinel score
        sampled_ids = []
        attention = []
        Beta = []
        
        # Initial hidden states
        states = None

        for i in range( max_len ):

            scores, states, atten_weights, beta = self.decoder( V, v_g, captions, states ) 
            predicted = scores.max( 2 )[ 1 ] # argmax
            captions = predicted
            
            # Save sampled word, attention map and sentinel at each timestep
            sampled_ids.append( captions )
            attention.append( atten_weights )
            Beta.append( beta )
        
        # caption: B x max_len
        # attention: B x max_len x 49
        # sentinel: B x max_len
        sampled_ids = torch.cat( sampled_ids, dim=1 )
        attention = torch.cat( attention, dim=1 )
        Beta = torch.cat( Beta, dim=1 )
        
        return sampled_ids, attention, Beta
