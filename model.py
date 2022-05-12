import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class Complex(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Complex, self).__init__()
        # @lw: init entities
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        # @lw: what is img?
        self.emb_e_img = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        # @lw: xavier with normal distribution
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):
        # @lw: torch.squeeze: Returns a tensor with all the dimensions of input of size 1 removed.
        # @lw: why use here?
        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img =  self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, self.emb_e_img.weight.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, self.emb_e_img.weight.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, self.emb_e_real.weight.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = torch.sigmoid(pred)

        return pred


class DistMult(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded*rel_embedded, self.emb_e.weight.transpose(1,0))
        pred = torch.sigmoid(pred)

        return pred



class ConvE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(ConvE, self).__init__()
        # @lw: init embeddings
        # @lw: num_entities = |E|, args.embedding_dim = the dimension of the embed
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        # @lw: the embed dim of rel is the same as the embed dim of entity
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        # @lw: set the dropout ratio
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        # @lw: what is the hideen layer?
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)
        # @lw: BCELoss = binary cross entropy loss
        # @lw: Q: the matrix manner of BCE?
        self.loss = torch.nn.BCELoss()
        # @lw: what is it?
        self.emb_dim1 = args.embedding_shape1
        # @lw: i dont get it.
        self.emb_dim2 = args.embedding_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias)
        # @lw: batch normalization, but why with parameters?
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        # @lw: REF: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # @lw:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        self.fc = torch.nn.Linear(args.hidden_size,args.embedding_dim)
        print(num_entities, num_relations)

    def init(self):
        """@lw: 
        normalize data in xavier (normal distri) manner
        """
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        # @lw: get the head and the relation embed, but what is the meaning of the dims?
        e1_embedded= self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        # @lw: stack the e1 and rel
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        # @lw: batch normalization
        stacked_inputs = self.bn0(stacked_inputs)
        # @lw: drop out the stacks
        x= self.inp_drop(stacked_inputs)
        # @lw: conv operation
        x= self.conv1(x)
        # @lw: batch normalization
        x= self.bn1(x)
        # @lw: relu the feature maps, only keep the values which are greater than 0
        # @lw: REF: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        x= F.relu(x)
        # @lw: droput the feature maps
        x = self.feature_map_drop(x)
        # @lw: shape the feature maps into a vector
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred


# Add your own model here

class MyModel(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)

        # Add your model function here
        # The model function should operate on the embeddings e1 and rel
        # and output scores for all entities (you will need a projection layer
        # with output size num_relations (from constructor above)

        # generate output scores here
        prediction = torch.sigmoid(output)

        return prediction
