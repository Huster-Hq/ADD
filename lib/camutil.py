import torch
import torch.nn.functional as F


def _refine_cams(ref_mod, images, cams,  orig_size):
    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    return refined_label

def refine_cams_with_bkg(ref_mod=None, images=None, cams=None, cfg=None,  down_scale=2):
    b,_,h,w=cams.shape
    _,_,h1,w1 = images.shape

    _images = F.interpolate(images, size=[h1//down_scale,w1//down_scale], mode="bilinear", align_corners=False)
    
    bkg_h = torch.ones(size=(b,1,h,w))*cfg.high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b,1,h,w))*cfg.low_thre
    bkg_l = bkg_l.to(cams.device)

    refined_label_h = torch.ones(size=(b, h, w)) * cfg.ignore_index
    refined_label_h = refined_label_h.to(cams.device)
    refined_label_l = refined_label_h.clone()
    
    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)

    for idx, _ in enumerate(images):
        valid_cams_h = cams_with_bkg_h[idx, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = cams_with_bkg_l[idx, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams_h, orig_size=(h,w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams_l, orig_size=(h,w))
        
        refined_label_h[idx, :, :] = _refined_label_h[0, :, :]
        refined_label_l[idx, :, :] = _refined_label_l[0, :, :]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = cfg.ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0
    return refined_label

# The key component: Semantic Relation Generation (SRG) module
def SR_generation(cam_label1,cam_label2, ignore_index=255,confuse_value=0.5):
    b, h, w = cam_label1.shape
    _cam_label1 = cam_label1.reshape(b, 1, -1)
    _cam_label_rep1 = _cam_label1.repeat([1, _cam_label1.shape[-1], 1])
    _cam_label2 = cam_label2.reshape(b, 1, -1)
    _cam_label_rep2 = _cam_label2.repeat([1, _cam_label2.shape[-1], 1])
    _cam_label_rep2_t = _cam_label_rep2.permute(0,2,1)
    aff_label = (_cam_label_rep1 == _cam_label_rep2_t).type(torch.long)
    
    for i in range(b):
        aff_label[i, :, _cam_label_rep1[i, 0, :]==ignore_index] = confuse_value
        aff_label[i, _cam_label_rep2[i, 0, :]==ignore_index, :] = confuse_value
    return aff_label