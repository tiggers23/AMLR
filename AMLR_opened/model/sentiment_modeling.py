from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import timm

from transformers import BertModel
from torchcrf import CRF


def flatten(x):
    if len(x.size()) == 2:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        return x.view([batch_size * seq_length])
    elif len(x.size()) == 3:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        hidden_size = x.size()[2]
        return x.view([batch_size * seq_length, hidden_size])
    else:
        raise Exception()


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, dropout2=False, attn_type='softmax'):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        if dropout2:
            # self.dropout2 = nn.Dropout(dropout2)
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout2)
        else:
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model)

        if n_head > 1:
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, attn_mask=None, dec_self=False):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        if hasattr(self, 'dropout2'):
            q = self.dropout2(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, attn_mask=attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        if hasattr(self, 'fc'):
            output = self.fc(output)

        if hasattr(self, 'dropout'):
            output = self.dropout(output)

        if dec_self:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1, attn_type='softmax'):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        if attn_type == 'softmax':
            self.attn_type = nn.Softmax(dim=2)
            # self.softmax = BottleSoftmax()
        else:
            self.attn_type = nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None, stop_sig=False):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            # attn = attn.masked_fill(attn_mask, -np.inf)
            attn = attn.masked_fill(attn_mask, -1e6)

        attn = self.attn_type(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class ViT(nn.Module):
    def __init__(self, model_name="vit_large_patch16_224_in21k", pretrained=True):
        super(ViT, self).__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        # Others variants of ViT can be used as well
        '''
        1 --- 'vit_small_patch16_224'
        2 --- 'vit_base_patch16_224'
        3 --- 'vit_large_patch16_224',
        4 --- 'vit_large_patch32_224'
        5 --- 'vit_deit_base_patch16_224'
        6 --- 'deit_base_distilled_patch16_224',
        '''

        # Change the head depending of the dataset used
        self.vit.head = nn.Identity()

    def forward(self, x):

        x = self.vit.patch_embed(x)

        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        if self.vit.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.vit.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)

        return x


#This MSFMNER is the AMLR model
class MSFMNER(nn.Module):
    def __init__(self, args, config):
        super(MSFMNER, self).__init__()

        self.text_encoder = BertModel.from_pretrained(pretrained_model_name_or_path="bert-base-uncased",
                                               output_hidden_states=True)
        self.vit = ViT("vit_base_patch16_224")
        self.imgLa2text_aspect = MultiHeadAttention(args.multi_head_num, 768, 768, 768)
        self.ner_span_affine = nn.Linear(768, 5)
        self.crf_13 = CRF(13, batch_first=True)
        self.ner_crf_13_classify = nn.Linear(768, 13)
        self.span_2 = nn.Linear(2 * 768, 768)
        self.span_3 = nn.Linear(3 * 768, 768)
        self.span_4 = nn.Linear(4 * 768, 768)
        self.span_5 = nn.Linear(5 * 768, 768)
        self.span_6 = nn.Linear(6 * 768, 768)
        self.loss_fn = torch.nn.CrossEntropyLoss()


    def fix_params(self):
        self.pretrain_model.eval()
        for param in self.pretrain_model.parameters():
            param.requires_grad = False

    def flatten(self, x):
        if len(x.size()) == 2:
            batch_size = x.size()[0]
            seq_length = x.size()[1]
            return x.view([batch_size * seq_length])
        elif len(x.size()) == 3:
            batch_size = x.size()[0]
            seq_length = x.size()[1]
            hidden_size = x.size()[2]
            return x.view([batch_size * seq_length, hidden_size])
        else:
            raise Exception()

    def flatten_emb_by_token_level_span_mask(self, emb, emb_mask):
        batch_size = emb.size()[0]
        seq_length = emb.size()[1]
        flat_emb = self.flatten(emb)
        flat_emb_mask = emb_mask.view([batch_size * seq_length])
        return flat_emb[flat_emb_mask.nonzero().squeeze(), :]

    def flatten_label_by_token_level_span_mask(self, span_label, emb_mask):
        batch_size = span_label.size()[0]
        seq_length = span_label.size()[1]
        flat_emb = self.flatten(span_label)
        flat_emb_mask = emb_mask.view([batch_size * seq_length])
        return flat_emb[flat_emb_mask.nonzero().squeeze()]

    def span_feature_generator(self, feature, mask, span_start, span_end, max_seq_length):

        feature1 = feature
        for i in range(max_seq_length, span_start.shape[-1]):

            if span_end[0, i].item() - span_start[0, i].item() == 1:
                sum_feature_2 = feature[:, span_start[0, i], :]
                for j in range(1, span_end[0, i].item() - span_start[0, i].item() + 1):
                    sum_feature_2 = torch.cat((sum_feature_2, feature[:, span_start[0, i] + j, :]), dim=1)
                sum_feature_2 = self.span_2(sum_feature_2)
                feature1 = torch.cat((feature1, sum_feature_2.reshape(span_start.shape[0], -1, feature.shape[2])),
                                     dim=1)

            if span_end[0, i].item() - span_start[0, i].item() == 2:
                sum_feature_3 = feature[:, span_start[0, i], :]
                for j in range(1, span_end[0, i].item() - span_start[0, i].item() + 1):
                    sum_feature_3 = torch.cat((sum_feature_3, feature[:, span_start[0, i] + j, :]), dim=1)
                sum_feature_3 = self.span_3(sum_feature_3)
                feature1 = torch.cat((feature1, sum_feature_3.reshape(span_start.shape[0], -1, feature.shape[2])),
                                     dim=1)

            if span_end[0, i].item() - span_start[0, i].item() == 3:
                sum_feature_4 = feature[:, span_start[0, i], :]
                for j in range(1, span_end[0, i].item() - span_start[0, i].item() + 1):
                    sum_feature_4 = torch.cat((sum_feature_4, feature[:, span_start[0, i] + j, :]), dim=1)
                sum_feature_4 = self.span_4(sum_feature_4)
                feature1 = torch.cat((feature1, sum_feature_4.reshape(span_start.shape[0], -1, feature.shape[2])),
                                     dim=1)

            if span_end[0, i].item() - span_start[0, i].item() == 4:
                sum_feature_5 = feature[:, span_start[0, i], :]
                for j in range(1, span_end[0, i].item() - span_start[0, i].item() + 1):
                    sum_feature_5 = torch.cat((sum_feature_5, feature[:, span_start[0, i] + j, :]), dim=1)
                sum_feature_5 = self.span_5(sum_feature_5)
                feature1 = torch.cat((feature1, sum_feature_5.reshape(span_start.shape[0], -1, feature.shape[2])),
                                     dim=1)

            if span_end[0, i].item() - span_start[0, i].item() == 5:
                sum_feature_6 = feature[:, span_start[0, i], :]
                for j in range(1, span_end[0, i].item() - span_start[0, i].item() + 1):
                    sum_feature_6 = torch.cat((sum_feature_6, feature[:, span_start[0, i] + j, :]), dim=1)
                sum_feature_6 = self.span_6(sum_feature_6)
                feature1 = torch.cat((feature1, sum_feature_6.reshape(span_start.shape[0], -1, feature.shape[2])),
                                     dim=1)

        return feature1

    def forward(self, args, label_ids_noSpan, input_ids_noSpan, raw_image_data, mask_noSpan,
                     segment_ids_noSpan, span_ner_labels, span_token_level_mask, span_start, span_end, label_ids_crf13, train=True):

        image_embeds = self.vit(raw_image_data)

        as_multi_out_orig = self.text_encoder(input_ids_noSpan,
                                                 attention_mask=mask_noSpan
                                                 )

        as_multi_out_before_cross_attention = self.span_feature_generator(as_multi_out_orig[0],
                                                                          mask_noSpan, span_start, span_end,
                                                                          args.max_seq_length)
        as_multi_out, _ = self.imgLa2text_aspect(as_multi_out_before_cross_attention, image_embeds,
                                                 image_embeds)
        as_multi_out_masked = self.flatten_emb_by_token_level_span_mask(as_multi_out, span_token_level_mask)

        ner_span_outputs = self.ner_span_affine(as_multi_out_masked)

        span_ner_labels_masked = self.flatten_label_by_token_level_span_mask(span_ner_labels,
                                                                             span_token_level_mask)

        if train:

            ner_span_crf13 = self.ner_crf_13_classify(as_multi_out_orig[0])
            ner_text_loss = -self.crf_13(ner_span_crf13, label_ids_crf13, mask=mask_noSpan)

            ner_neg_loss = 0
            ner_pos_loss = 0
            for i in range(len(span_ner_labels_masked)):

                if span_ner_labels_masked[i] == 0:
                    ner_neg_loss += self.loss_fn(ner_span_outputs[i], span_ner_labels_masked[i])
                else:
                    ner_pos_loss += self.loss_fn(ner_span_outputs[i], span_ner_labels_masked[i])

            weight = 10
            loss = ner_neg_loss + weight*ner_pos_loss + ner_text_loss

            return loss


        else:

            return ner_span_outputs, span_ner_labels_masked


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, dropout2=False, attn_type='softmax'):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        if dropout2:
            # self.dropout2 = nn.Dropout(dropout2)
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout2)
        else:
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model)

        if n_head > 1:
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, attn_mask=None, dec_self=False):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        if hasattr(self, 'dropout2'):
            q = self.dropout2(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

            # add by lep
            attn_mask = attn_mask.view(-1, 1, attn_mask.size()[2])  # (n*b) x lv x dv
            # attn_mask = attn_mask.unsqueeze()

        output, attn = self.attention(q, k, v, attn_mask=attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        if hasattr(self, 'fc'):
            output = self.fc(output)

        if hasattr(self, 'dropout'):
            output = self.dropout(output)

        if dec_self:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output + residual)

        return output, attn
