"""
This file contains the implementation of the base class for the SSL regressor model.
"""
import torch
import torch.nn as nn

class ssl_mospred_model(nn.Module):
    """
    Base class for the SSL regressor model
    """
    def __init__(
        self, 
        ssl_model=None,
        num_layers=None,
        args=None,
        feat_dim=None,
    ):
        """
        Initialize the model
        """
        super(ssl_mospred_model, self).__init__()
        self.ssl_model = ssl_model
        
        
        dim = feat_dim
        
        ##move this to config
        if num_layers == 1:
            
            if args.use_cer:
                dim = 68
            
            if args.use_lang:
                dim += 32
            
            if args.use_task:
                dim += 32
            
        elif num_layers == 3:
            
            if args.use_cer:
                dim = 68
            
            if args.use_lang:
                dim += 32
            
            self.feat_learner = nn.Sequential(
                nn.Linear(dim, dim//2),
                nn.ReLU(),
                nn.Linear(dim//2, dim//2),
                nn.ReLU(),
            )
            dim = dim//2
                    
        else:
            raise NotImplementedError()
        
        self.linear = nn.Linear(dim, 1)
        
        self.args = args
        if args.use_cer:
            self.cer_embed = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 4),
                nn.ReLU(),
            )
            self.feat_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(feat_dim, 64),
            )
            
        if args.use_lang:
            self.lang_embed = nn.Embedding(7, 32)
        
        if args.use_task:
            self.task_embed = nn.Embedding(2, 32)
        
        
            
            
    
    def get_padding_mask(self, x, feats, lengths):
        """
        Get the padding mask for the input
        """
        max_length = feats.shape[1]
        num_frames = round(x.shape[-1]/feats.shape[1])
        ssl_lengths = [int(l/(num_frames)) for l in lengths]
        ssl_lengths = torch.LongTensor(ssl_lengths)
        mask = (torch.arange(max_length).expand(len(ssl_lengths), max_length) < ssl_lengths.unsqueeze(1)).float()
        return mask.to(self.args.device)
    
    def handle_cer_embed(self, feats, cer):
        """
        CER embedding
        """
        if not self.args.use_cer:
            return feats
        feats = self.feat_proj(feats)
        cer = self.cer_embed(cer[:, None])
        feats = torch.cat([feats, cer], -1)
        return feats

    def handle_lang_embed(self, feats, lang):
        """
        Language embedding
        """
        if not self.args.use_lang:
            return feats
        lang = self.lang_embed(lang)
        feats = torch.cat([feats, lang], -1)
        return feats

    def handle_task_embed(self, feats, task):
        """
        Task embedding
        """
        if not self.args.use_task:
            return feats
        task = self.task_embed(task)
        feats = torch.cat([feats, task], -1)
        return feats

    def forward(self, x, lengths, cer_data=None, lang_data=None, mc_data=None, task_data=None):
        """
        Forward pass
        """
        feats = self.ssl_model(x)["hidden_states"][-1]
        mask = self.get_padding_mask(x, feats, lengths)
        feats = feats * mask.unsqueeze(-1)
        feats = feats.sum(1)/mask.sum(-1).unsqueeze(-1)
        feats = self.handle_cer_embed(feats, cer_data)
        feats = self.handle_lang_embed(feats, lang_data)
        feats = self.handle_task_embed(feats, task_data)
        if hasattr(self, 'feat_learner'):
            feats = self.feat_learner(feats)
        feats = self.linear(feats)
        return feats.float()

    def regression_loss(self, output, target):
        """
        Regression loss
        """
        return nn.MSELoss()(output.squeeze(), target.float().squeeze())
    