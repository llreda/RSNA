from typing import Sequence

from mmengine.model import BaseModule, BaseModel
from mmengine.structures import BaseDataElement


from mm_custom.registry import MODELS
import torch
import torch.nn.functional as F
import torch.nn as nn


from mmdet.models.layers import Mask2FormerTransformerDecoder

from .atd_head_utils import ClipSinePositionalEncoding



# mask2former 
@MODELS.register_module()
class AtdHeadALLMask(BaseModule):
    def __init__(self, 
                 transformer_decoder_cfg_masks,
                 transformer_decoder_cfg_extra,
                 clip_num,
                 in_channels,
                 feat_channels,  
                 mask_channels,  # 既是pixel_decode的输出通道数，也是最后query的通道数
                 mask_weight,
                 metainfo=None,
                 init_cfg=None):
        super().__init__(init_cfg)


        self.transformer_decoder_cfg_masks = transformer_decoder_cfg_masks
        self.transformer_decoder_cfg_extra = transformer_decoder_cfg_extra

        self.clip_num = clip_num

        self.metainfo = metainfo

        self.feat_channels = feat_channels
        self.mask_channels = mask_channels
        self.in_channels = in_channels
        self.mask_weight = mask_weight

        self.mask_branch_names = metainfo['mask_names']
        self.extra_branch_names = metainfo['extra_names']

        self.mask_branch_cls_heads = nn.ModuleList()
        self.extra_branch_cls_heads = nn.ModuleList()

        self.class_names = metainfo['class_names']
        self.out_channels = metainfo['out_channels']
        class2channels = {self.class_names[i]:self.out_channels[i] for i in range(len(self.class_names))}

        for name in self.mask_branch_names:
            self.mask_branch_cls_heads.append(
                nn.Sequential(
                    nn.Linear(self.feat_channels, class2channels[name]),
                )
            )

        for name in self.extra_branch_names:
            self.extra_branch_cls_heads.append(
                nn.Sequential(
                    nn.Linear(self.feat_channels, class2channels[name]),
                )
            )

        self.image_level_class_names = metainfo['image_level_class_names']
        self.binary_class_names = metainfo['binary_class_names']

        self.image_level_heads = nn.ModuleList()
        for i in range(len(self.image_level_class_names)):
            self.image_level_heads.append(
                nn.Sequential(
                    nn.Linear(self.feat_channels, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 2)
                )
            )

        # 从query feat的特征空间转到mask的特征空间，与mask_feat 的维度相同，从而预测一个mask
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, mask_channels))  
        
        if self.in_channels[0] != self.mask_channels:
            self.mask_project = nn.Conv2d(in_channels[0], self.mask_channels, kernel_size=1)

        self.mask_branch_num_queries = len(self.mask_branch_names)
        self.extra_branch_num_queries = len(self.extra_branch_names)

        
        self.loss_func_cls = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.05) 
        self.loss_func_image_level = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.) 
        self.loss_func_any_injury = nn.NLLLoss(reduction='none')

        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=1.0
        )

        self.loss_dice = MODELS.build(loss_dice)
        self.loss_mask = nn.BCEWithLogitsLoss()


        self.init_mask_branch_decoder_layers()

        self.init_extra_branch_decoder_layers()



    def init_mask_branch_decoder_layers(self):
        
        transformer_decoder = self.transformer_decoder_cfg_masks

        self.mask_branch_num_decoder_layers = transformer_decoder.num_layers
        self.mask_branch_num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads

        self.mask_branch_transformer_decoder = Mask2FormerTransformerDecoder(**transformer_decoder)
        
        feat_channels = self.mask_branch_transformer_decoder.embed_dims

        self.mask_branch_query_embed = nn.Embedding(self.mask_branch_num_queries, feat_channels)
        self.mask_branch_query_feat = nn.Embedding(self.mask_branch_num_queries, feat_channels)

        # positional encoding
        positional_encoding=dict(num_feats=(self.feat_channels//2), normalize=True)  # SinePositionalEncoding    
        self.mask_branch_clip_decoder_positional_encoding = ClipSinePositionalEncoding(**positional_encoding)
        self.init_mask_branch()


    def init_extra_branch_decoder_layers(self):
        
        transformer_decoder = self.transformer_decoder_cfg_extra

        self.extra_branch_num_decoder_layers = transformer_decoder.num_layers
        self.extra_branch_num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads


        self.extra_branch_transformer_decoder = Mask2FormerTransformerDecoder(**transformer_decoder)

        feat_channels = self.extra_branch_transformer_decoder.embed_dims

        self.extra_branch_query_embed = nn.Embedding(self.extra_branch_num_queries, feat_channels)
        self.extra_branch_query_feat = nn.Embedding(self.extra_branch_num_queries, feat_channels)

        # positional encoding
        positional_encoding=dict(num_feats=(self.feat_channels//2), normalize=True)  # SinePositionalEncoding    

        self.extra_branch_clip_decoder_positional_encoding = ClipSinePositionalEncoding(**positional_encoding)
        self.init_extra_branch()


    def init_mask_branch(self) -> None:
        
        for p in self.mask_branch_transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                

    def init_extra_branch(self):
        for p in self.extra_branch_transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    


    def forward_mask_branch(self, batch_inputs, mode):
        features = batch_inputs['inputs'][0]

        mask_features = features[0]
        features = features[1]
        

        BT, C, H, W = features.shape # [BT, C, H, W]

        T = self.clip_num
        B = BT // T


        features = features.view(B, T, C, H, W)
        decoder_inputs = features.permute(0, 1, 3, 4, 2).flatten(1, 3) # [B, T*H*W, C]
        
        decoder_positional_encoding = self.mask_branch_clip_decoder_positional_encoding(features) # [B, T, C, H, W]
        decoder_positional_encoding = decoder_positional_encoding.permute(0, 1, 3, 4, 2).flatten(1, 3) # [B, T*H*W, C]

        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.mask_branch_query_feat.weight.unsqueeze(0).repeat((B, 1, 1))
        query_embed = self.mask_branch_query_embed.weight.unsqueeze(0).repeat((B, 1, 1))

        cls_pred_list = []  # q, b, c
        mask_pred_list = []  # bt, q, h, w

        cls_pred, mask_pred, attn_mask = self._forward_mask_head(
            query_feat, mask_features, features.shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.mask_branch_num_decoder_layers):
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            layer = self.mask_branch_transformer_decoder.layers[i]

            query_feat = layer(
                query=query_feat,
                key=decoder_inputs,
                value=decoder_inputs,
                query_pos=query_embed,
                key_pos=decoder_positional_encoding,
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None                
            )

            cls_pred, mask_pred, attn_mask = self._forward_mask_head(
            query_feat, mask_features, features.shape[-2:])
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
        

        if mode == 'loss':
            target_dic = batch_inputs['target_dic']
            weight_dic = batch_inputs['weight_dic']
            mask_dic = batch_inputs['mask_dic']

            loss_dict = self.loss_mask_branch(cls_pred_list, target_dic, mask_pred_list, mask_dic, weight_dic)
            
            return loss_dict

        else:
            # data_samples = self.predict(cls_pred_list[-1])
            return cls_pred_list[-1]
        


    def forward_extra_branch(self, batch_inputs, mode):
        features = batch_inputs['inputs'][1]

        features = features[1]   # 取一个尺度的特征，一个query，不要mask，直接输出
        

        BT, C, H, W = features.shape # [BT, C, H, W]

        T = self.clip_num
        B = BT // T
        

        image_level_head_outs = self._forward_image_level(features)  # [BT,2]


        features = features.view(B, T, C, H, W)
        decoder_inputs = features.permute(0, 1, 3, 4, 2).flatten(1, 3) # [B, T*H*W, C]
        
        decoder_positional_encoding = self.extra_branch_clip_decoder_positional_encoding(features) # [B, T, C, H, W]
        decoder_positional_encoding = decoder_positional_encoding.permute(0, 1, 3, 4, 2).flatten(1, 3) # [B, T*H*W, C]

        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.extra_branch_query_feat.weight.unsqueeze(0).repeat((B, 1, 1))
        query_embed = self.extra_branch_query_embed.weight.unsqueeze(0).repeat((B, 1, 1))

        cls_pred_list = []  # q, b, c
        
        
        for i in range(self.extra_branch_num_decoder_layers):
            
            layer = self.extra_branch_transformer_decoder.layers[i]

            query_feat = layer(
                query=query_feat,
                key=decoder_inputs,
                value=decoder_inputs,
                query_pos=query_embed,
                key_pos=decoder_positional_encoding,
                # cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None                
            )

            cls_pred = self._forward_extra_head(query_feat)
            cls_pred_list.append(cls_pred)
           
        
        if mode == 'loss':
            target_dic = batch_inputs['target_dic']
            weight_dic = batch_inputs['weight_dic']
          
            image_level_target_dic = batch_inputs['image_level_target_dic']

            loss_dict = self.loss_extra_branch(cls_pred_list, target_dic, image_level_head_outs,image_level_target_dic, weight_dic)
            
            return loss_dict

        else:

            return cls_pred_list[-1]
        

    
    def _forward_image_level(self, features):
        
        # feat: [BT, C, H, W]

        BT, C, H, W = features.shape

        T = self.clip_num
        B = BT // T


        feat = F.adaptive_avg_pool2d(features, output_size=(1, 1)).squeeze(-1).squeeze(-1) # [BT, C]
        
        outputs = []    

        for i, name in enumerate(self.image_level_class_names):
            output = self.image_level_heads[i](feat)    # [BT, 2]
            outputs.append(output)

        return outputs
                       

    def _forward_extra_head(self, decoder_out):
        decoder_out = self.extra_branch_transformer_decoder.post_norm(decoder_out)  # b q c
        # shape (num_queries, batch_size, c)
        cls_preds = []
        for i in range(self.extra_branch_num_queries):
            head_out = self.extra_branch_cls_heads[i](decoder_out[:, i, :])  # b * c
            cls_preds.append(head_out)
        
        return cls_preds
 

    def _forward_mask_head(self, decoder_out, mask_feature, attn_mask_target_size):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.mask_branch_transformer_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        
        cls_preds = []

        for i in range(self.mask_branch_num_queries):
            head_out = self.mask_branch_cls_heads[i](decoder_out[:, i, :])  # b * c
            cls_preds.append(head_out)

        mask_embed = self.mask_embed(decoder_out)
       
        mask_embed = mask_embed.unsqueeze(1).repeat(1, self.clip_num, 1, 1).flatten(0, 1)

        # shape (bt, q, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)    # [batch_size, num_queries, h, w]
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)

        num_queries, H, W = attn_mask.shape[-3:]

        attn_mask = attn_mask.view(-1, self.clip_num, num_queries, H, W).permute(0, 2, 1, 3, 4) # [B, num_queries, T, H, W]

        # [B, num_queries, T*H*W]
        attn_mask = attn_mask.flatten(2)

        # [B, num_queries, T*H*W] -> [B*num_heads, num_queries, T*H*W]
        attn_mask = attn_mask.unsqueeze(1).repeat((1, self.mask_branch_num_heads, 1, 1)).flatten(0, 1)
        
        attn_mask = attn_mask.sigmoid() < 0.5

        return cls_preds, mask_pred, attn_mask


    def loss_extra_branch(self, cls_pred_list, target_dic, image_level_head_outs, image_level_target_dic, weight_dic):
        loss_dict = {}

        for layer in range(len(cls_pred_list)):
            
            cls_pred = cls_pred_list[layer]

            for i, class_name in enumerate(self.extra_branch_names):
                
                output = cls_pred[i] # [B, C]

                target = target_dic[class_name].float() # [B, C]
                weight = weight_dic[class_name].float() # [B]
                
                loss = self.loss_func_cls(output, target)   # [B]

                loss_dict[f'{class_name}_cls_loss_layer{layer}'] = (loss * weight).mean()
            
        
        for i, name in enumerate(self.image_level_class_names):
            output = image_level_head_outs[i]         # [BT, 2]
            target = image_level_target_dic[name]  # [B, clip_num]
            B, T = target.shape

            weight = weight_dic[self.binary_class_names[i]]

            loss = self.loss_func_image_level(output, target.view(-1).long())

            loss = loss.view(B, T).mean(dim=1)

            loss_dict[f'{name}_ce_loss'] = (loss * weight).mean()
        
        return loss_dict
 
   

    def loss_mask_branch(self, cls_preds, target_dic, mask_preds, mask_dic, weight_dic):

        loss_dict = {}

        for layer in range(len(cls_preds)):
            if layer == 0:
                continue
            
            cls_pred = cls_preds[layer]

            for i, class_name in enumerate(self.mask_branch_names):
                
                output = cls_pred[i] # [B, C]

                target = target_dic[class_name].float() # [B, C]
                weight = weight_dic[class_name].float() # [B]
                
                loss = self.loss_func_cls(output, target)   # [B]

                loss_dict[f'{class_name}_cls_loss_layer{layer}'] = (loss * weight).mean()
            
            
            ## any injury loss
            healthy_probs = []
            for i, class_name in enumerate(self.mask_branch_names):
                probs = cls_pred[i].softmax(dim=1)
                healthy_prob = probs[:, 0:1]
                healthy_probs.append(healthy_prob)

            healthy_probs = torch.cat(healthy_probs, dim=1) # [b, 5]
            any_injury_probs = (1 - healthy_probs).max(dim=1).values

            any_injury_probs = torch.stack([1-any_injury_probs, any_injury_probs], dim=1)

            any_injury_ce_loss = self.loss_func_any_injury(torch.log(any_injury_probs), target_dic['any_injury'][:, 1].long())
            weight = weight_dic['any_injury']

            loss_dict[f'any_injury_ce_loss_layer{layer}'] = (any_injury_ce_loss * weight).mean()
        


        gt_masks = []
        for name in (self.mask_branch_names):
            gt_masks.append(mask_dic[name])
        
        gt_masks = torch.stack(gt_masks, dim=2) # [B, T, num_queries, H, W]
        gt_masks = gt_masks.flatten(0, 1).float() # [BT, num_queries, H, W] 

        for layer in range(len(mask_preds)):
            mask_pred = mask_preds[layer] # bt, q, h, w
            
            mask_pred = F.interpolate(
                mask_pred,
                size=gt_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss_dict[f'mask_loss_layer{layer}'] = self.loss_mask(mask_pred, gt_masks) * self.mask_weight      

            loss_dict[f'dice_loss_layer{layer}'] = self.loss_dice(mask_pred, gt_masks) * self.mask_weight

        return loss_dict        
    


    def forward(self, batch_inputs, mode):
        if mode == 'loss':
            loss_dict = self.forward_mask_branch(batch_inputs, mode)
            loss_dict2 = self.forward_extra_branch(batch_inputs, mode)
            loss_dict.update(loss_dict2)
            return loss_dict
        
        else:
            preds1 = self.forward_mask_branch(batch_inputs, mode) # preds1 里是四个的预测 [b*2, b*3, b*3, b*3]
            preds2 = self.forward_extra_branch(batch_inputs, mode) # preds2 是一个预测  [b*2]
            
            preds = {}
            for i, name in enumerate(self.mask_branch_names):
                preds[name] = preds1[i]
            for i, name in enumerate(self.extra_branch_names):
                preds[name] = preds2[i]

            return self.predict(preds)



    def predict(self, outputs):

        batch_pred_probs = {}

        for name, pred in outputs.items():
            pred_prob = torch.softmax(pred, dim=1)
            batch_pred_probs[name] = pred_prob
            batch_size = pred_prob.shape[0]


        data_samples = [BaseDataElement() for i in range(batch_size)]

        for batch in range(batch_size):
            pred_prob_dic = {}

            for name, pred_prob in batch_pred_probs.items():
                pred_prob_dic[name] = pred_prob[batch]
            

            data_samples[batch].set_data(
                {
                    'pred_prob_dic': pred_prob_dic,
                }
            )


        return data_samples



